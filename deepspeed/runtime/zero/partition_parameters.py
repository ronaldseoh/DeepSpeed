import os
import torch
from enum import Enum
import itertools
from deepspeed.runtime.zero.linear import LinearModuleForZeroStage3, LinearFunctionForZeroStage3
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.utils import log_dist

param_count = 0

def print_rank_0(message, debug=False, force=False):
    if torch.distributed.get_rank() == 0 and (debug or force):
        print(message)


class ZeroParamType(Enum):

    #same as regular pytorch parameters
    NORMAL = 1

    #parameters are partitioned across data parallel process
    PARTITIONED = 2

    #the parameter is held with a unique process rank
    #and is not available on all other process
    REMOTE = 3


class ZeroParamStatus(Enum):
    #parameters are fully present and ready for use on all processes
    AVAILABLE = 1

    #parameters are either partitioned or remote in some or all process
    NOT_AVAILABLE = 2

    #parameters are being gathered.
    INFLIGHT = 3


_orig_torch_empty = torch.empty


def empty_cuda_tensor(*size, **kwargs):
    kwargs['device'] = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    return _orig_torch_empty(*size, **kwargs).half()


def new_cuda_tensor(cls, *args):
    device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    return torch.ones((1, 1), device=device).new_empty(*args).half()


reuse_buffers = False
empty_buffers = {}


#Inserts _post_init_method at the end of init method
#for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self, zero_modules=False, dtype=None):
        self.zero_modules = zero_modules
        self.dtype = dtype
    def __enter__(self):
        # torch.Tensor.__new_original__ = torch.Tensor.__new__
        # torch.old_empty = torch.empty
        # torch.Tensor.__new__ = new_gpu_tensor
        # torch.empty = empty_gpu_tensor

        def partition_after(f):
            def wrapper(module, *args, **kwargs):
                print_rank_0(f'Before initializing {module.__class__.__name__}',
                             force=False)
                f(module, *args, **kwargs)
                self._post_init_method(module)
                print_rank_0(
                    f'After initializing followed by post init for {module.__class__.__name__}',
                    force=False)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)

        def register_external_parameter(cls, name, param):
            if not hasattr(cls, '_external_params'):
                cls._external_params = {}

            assert isinstance(param,torch.nn.Parameter), "param is not a torch.nn.parameter"
            cls._external_params[name] = param

        def external_parameters(cls):
            if not hasattr(cls, '_external_params'):
                cls._external_params = {}
            return cls._external_params.items()

        def all_parameters(cls):
            return itertools.chain(cls.named_parameters(cls,
                                                        recurse=False),
                                   external_parameters(cls))

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _enable_class(subclass)

        #holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        torch.Tensor.__new__ = new_cuda_tensor
        torch.empty = empty_cuda_tensor

        torch.nn.modules.module.Module.ds_register_external_parameter = classmethod(
            register_external_parameter)
        torch.nn.modules.module.Module.ds_external_parameters = classmethod(
            external_parameters)
        torch.nn.modules.module.Module.ds_all_parameters = classmethod(all_parameters)

        if self.zero_modules:
            self.linear_bk = torch.nn.functional.linear
            torch.nn.functional.linear = LinearFunctionForZeroStage3.apply

            # torch.nn.modules.linear.Linear_bk = torch.nn.modules.linear.Linear
            # torch.nn.Linear_bk = torch.nn.Linear
            # torch.nn.modules.linear.Linear = LinearModuleForZeroStage3
            # torch.nn.Linear = LinearModuleForZeroStage3

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass

        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.empty = _orig_torch_empty

        if self.zero_modules:
            torch.nn.functional.linear = self.linear_bk
            # torch.nn.modules.linear.Linear = torch.nn.modules.linear.Linear_bk
            # torch.nn.Linear = torch.nn.Linear_bk
            # delattr(torch.nn.modules.linear, 'Linear_bk')
            print_rank_0("Using ZeRO Linear")

        #delattr(torch.nn.modules.module.Module, 'ds_register_external_parameter')
        #delattr(torch.nn.modules.module.Module, 'ds_external_parameters')

    #To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass


#Replaces all parameters in module with Scattered Parameters
class ScatteredParameters(InsertPostInitMethodToModuleSubClasses):
    param_id = 0

    def __init__(self, ds_group=None, zero_modules=False, dtype=None, remote_device=None):
        super(ScatteredParameters, self).__init__(zero_modules=zero_modules, dtype=dtype)
        assert torch.distributed.is_initialized(), "Parameters cannot be scattered without initializing torch.distributed"
        self.ds_process_group = torch.distributed.group.WORLD if ds_group is None else ds_group
        self.rank = torch.distributed.get_rank(group=self.ds_process_group)
        self.world_size = torch.distributed.get_world_size(group=self.ds_process_group)
        
        #Local device is the device where the parameters are consumed
        #It is the device where parameters are fully instantiated using allgather
        self.local_device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))

        #Remote device is the device where parameter partiitons are stored
        #It can be same as local_device or it could be CPU or NVME.
        self.remote_device = self.local_device if remote_device is None else remote_device

    def _post_init_method(self, module):
        #see_memory_usage(f"Before converting parmas in {module.__class__.__name__}", force=False)
        print_rank_0(f'Converting Params in {module.__class__.__name__}', force=False)
        see_memory_usage(f"Before converting and partitioning parmas in {module.__class__.__name__}", force=False)
        global param_count
        for name, param in module.named_parameters(recurse=False):
            param_count += param.numel()
            if not hasattr(param, 'ds_id'):
                self._convert_to_deepspeed_param(param)
                print_rank_0(
                    f"Partitioning param with ds id {param.ds_id} and shape {param.data.shape}"
                )
                param.partition()
        see_memory_usage(f"Param count {param_count}. After converting and partitioning parmas in {module.__class__.__name__}", force=False)

    def _convert_to_deepspeed_param(self, param):

        # Partitioned, Normal, Remote
        param.ds_param_type = ZeroParamType.PARTITIONED

        # Replicated vs Partitioned vs Inflight
        param.ds_status = ZeroParamStatus.AVAILABLE

        #Stores the shape of the original tensor
        param.ds_shape = param.shape

        #Stores the number of elements in the original parmaeter without padding
        param.ds_numel = param.numel()

        #Stores the paritioned copy of the tensor
        param.ds_tensor = None

        #Keeps track of how many active sub-modules need this param at any given point in time
        param.ds_active_sub_modules = 0

        #If this flag is true, then the parameters are replicated throughput training
        #And only partitioned before the step
        param.ds_persist = False

        #DeepSped Param ID
        param.ds_id = ScatteredParameters.param_id
        ScatteredParameters.param_id += 1

        def all_gather(param_list=None, async_op=False, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)

        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param with id {cls.ds_id} type {cls.dtype} dev {cls.device} shape {cls.shape}"
            )
            if param_list is None:
                param_list = [cls]
            self._partition(param_list, has_been_updated=has_been_updated)

        def reduce_gradients_at_owner(param_list=None, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            print_rank_0(
                f"{'--'*hierarchy}----Reducing Gradients for param with ids {[param.ds_id for param in param_list]} to owner"
            )
            self._reduce_scatter_gradients(param_list)

        def partition_gradients(param_list=None, partition_buffers=None, hierarchy=0, accumulate=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param gradient with id {cls.ds_id}")
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers, torch.Tensor):
                    partition_buffers = [partition_buffers]
            self._partition_gradients(param_list, partition_buffers=partition_buffers, accumulate=accumulate)

        def aligned_size():
            return self._aligned_size(param)

        def padding_size():
            return self._padding_size(param)

        #Collectives for gathering and partitioning parameters
        param.all_gather = all_gather
        param.partition = partition

        #Collective for averaging gradients
        param.reduce_gradients_at_owner = reduce_gradients_at_owner
        param.partition_gradients = partition_gradients

        # Partitioning size utilities
        param.aligned_size = aligned_size
        param.padding_size = padding_size

    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.world_size
        return (self.world_size - remainder) if remainder else 0

    def _all_gather(self, param_list, async_op=False, hierarchy=None):
        handles = []
        all_gather_list = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if async_op:
                    handle = self._allgather_param(param,
                                                   async_op=async_op,
                                                   hierarchy=hierarchy)
                    param.ds_status = ZeroParamStatus.INFLIGHT  #if async_op else ZeroParamStatus.AVAILABLE
                    handles.append(handle)
                else:
                    all_gather_list.append(param)

        if not async_op:
            ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            for param in all_gather_list:
                param.ds_status = ZeroParamStatus.AVAILABLE
            return ret_value

        return handles

    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            #print_rank_0(f"Before Partitioning Param {param.ds_id}")
            #self._param_status(param)
            self._partition_param(param, has_been_updated=has_been_updated)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            #if param.ds_tensor is not None:
            #    assert id(param.data) == id(param.ds_tensor.data), \
            #    "After the parameters are initially partitioned, make sure we are not recreating the partition."
            #print_rank_0(f"After Partitioning Param {param.ds_id}")
            #self._param_status(param)

    def _partition_param(self, param, has_been_updated=False):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot parititon a param in flight"
        global reuse_buffers
        #print_rank_0(f"Param id {param.ds_id} status is {param.ds_status}")
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            print_rank_0(
                f"Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}",
                force=False)
            if reuse_buffers and (param.ds_numel == 16384 * 16384 * 4
                                  or param.ds_numel == 16384 * 16384):
                buffer = param.data
                print_rank_0(
                    "Returning buffer for param {param.ds_id} with numel {param.ds_numel} to empty buffers",
                    force=False)
                empty_buffers[id(buffer)] = buffer

            if param.ds_tensor is not None and not has_been_updated:
                
                    #param.data = param.ds_tensor.data
                    
                    #param.data does not store anything meaningful in partitioned state
                    param.data = torch.ones(1).half().to(param.device)
                    
                    # print_rank_0(
                    #     f"Partitioning available param {param.ds_id} type {param.dtype} shape {param.shape}"
                    # )
                    return
            
            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.world_size

            if param.ds_tensor is None:
                partitioned_tensor = torch.zeros(partition_size,
                                                 dtype=param.dtype,
                                                 device=self.remote_device)
                partitioned_tensor.requires_grad = False
                if self.remote_device == 'cpu':
                    partitioned_tensor = partitioned_tensor.pin_memory()

                param.ds_tensor = partitioned_tensor

            start = partition_size * self.rank
            end = start + partition_size

            one_dim_param = param.contiguous().view(-1)

            
            if start < param.ds_numel and end <= param.ds_numel:
                src_tensor = one_dim_param.narrow(
                    0,
                    start,
                    partition_size)
                
                param.ds_tensor.copy_(src_tensor)
                #partitioned_tensor = src_tensor.clone().detach().to(self.remote_device)

            else:
                # partitioned_tensor = torch.zeros(partition_size,
                #                                  dtype=param.dtype,
                #                                  device=self.remote_device )

                if start < param.ds_numel:
                    elements_to_copy = param.ds_numel - start
                    param.ds_tensor.narrow(0,
                                              0,
                                              elements_to_copy).copy_(
                                                  one_dim_param.narrow(
                                                      0,
                                                      start,
                                                      elements_to_copy))

            print(f"Remote device {self.remote_device}")
            
            #param.ds_tensor = partitioned_tensor
            
            #param.data = param.ds_tensor.data

            #param.data does not store anything meaningful in partitioned state
            param.data = torch.ones(1).half().to(param.device)

            print_rank_0(
                f"ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}"
            )

    def _param_status(self, param):
        if param.ds_tensor is not None:
            print_rank_0(
                f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned numel {param.ds_tensor.numel()}, data numel {param.data.numel()}"
            )
        else:
            print_rank_0(
                f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned ds_tensor {param.ds_tensor}, data numel {param.data.numel()}"
            )

    def _allgather_param(self, param, async_op=False, hierarchy=0):

        #self._param_status(param)
        #partition_size = param.data.numel()
        partition_size = param.ds_tensor.numel()
        
        tensor_size = partition_size * self.world_size
        aligned_param_size = self._aligned_size(param)
        assert tensor_size == aligned_param_size, f'param id {param.ds_id} aligned size {aligned_param_size} does not match tensor size {tensor_size}'

        global empty_buffers, reuse_buffers

        flat_tensor = None
        buffer_key = None
        if reuse_buffers:
            #print(f"{empty_buffers}")
            for key, t in empty_buffers.items():
                if t.numel() == param.ds_numel:
                    flat_tensor = t.view(-1)
                    buffer_key = key
                    print_rank_0(
                        f"Buffer reused for allgather of param {param.ds_id} with {param.ds_numel} elements",
                        force=False)
        if buffer_key:
            empty_buffers.pop(buffer_key)
            assert buffer_key not in empty_buffers, "Empty buffers contains the tensor after removing"

        print_rank_0(
            f"{'--'* hierarchy}---- Before allocating Allgather param with id {param.ds_id} and status {param.ds_status} Partition Size {partition_size} and data shape {param.ds_shape}"
        )
        if flat_tensor is None:
            flat_tensor = torch.zeros(aligned_param_size,
                                      dtype=param.dtype,
                                      device=param.device).view(-1)
            torch.cuda.synchronize()

        print_rank_0(
            f"{'--'* hierarchy}----Allgather param with id {param.ds_id} and status {param.ds_status} Partition Size {partition_size} and data shape {param.ds_shape}"
        )
        #        if not flat_tensor.numel() > 100000:
        #            replicated_tensor = flat_tensor.narrow(0,
        #                                                   0,
        #                                                   param.ds_numel).view(param.ds_shape)
        #            param.data = replicated_tensor.data
        #            return None
        partitions = []
        for i in range(self.world_size):
            partitions.append(flat_tensor.narrow(0, partition_size * i, partition_size))

            if i == torch.distributed.get_rank(group=self.ds_process_group):
                #partitions[i].copy_(param.data)
                partitions[i].copy_(param.ds_tensor)

        handle = torch.distributed.all_gather(partitions,
                                              partitions[self.rank],
                                              group=self.ds_process_group,
                                              async_op=async_op)

        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        #param.data = flat_tensor.data
        #print(f"........ID {param.ds_id} gathered and contains {param.shape} and gradient {param.grad}")

        return handle

    def _allgather_params(self, param_list, hierarchy=0):
        # for param in param_list:
        #     replicated_tensor = torch.empty(param.ds_shape, dtype=param.dtype, device=param.device)
        #     param.data = replicated_tensor.data
        # return None

        #partition_size = sum([param.data.numel() for param in param_list])
        partition_size = sum([param.ds_tensor.numel() for param in param_list])
        
        tensor_size = partition_size * self.world_size
        flat_tensor = torch.empty(tensor_size,
                                  dtype=param_list[0].dtype,
                                  device=self.local_device)
        flat_tensor.requres_grad=False
        partitions = []
        for i in range(self.world_size):
            start = partition_size * i

            partitions.append(flat_tensor.narrow(0, start, partition_size))

            if i == self.rank:
                offset = 0
                for param in param_list:
                    #param_numel = param.data.numel()
                    param_numel = param.ds_tensor.numel()
                    
                    #partitions[i].narrow(0, offset, param_numel).copy_(param.data)
                    partitions[i].narrow(0, offset, param_numel).copy_(param.ds_tensor.data)
                    
                    offset += param_numel
        
        torch.distributed.all_gather(partitions,
                                     partitions[self.rank],
                                     group=self.ds_process_group,
                                     async_op=False)
        param_offset = 0

        for param in param_list:

            #param_partition_size = param.data.numel()
            param_partition_size = param.ds_tensor.numel()

            param_size = param.ds_numel
            replicated_tensor = torch.empty(param.ds_shape,
                                            dtype=param.dtype,
                                            device=self.local_device)

            for i in range(self.world_size):

                start = i * partition_size

                param_start = i * param_partition_size

                if param_start < param_size:
                    numel_to_copy = min(param_size - param_start, param_partition_size)

                    part_to_copy = partitions[i].narrow(0, param_offset, numel_to_copy)

                    replicated_tensor.view(-1).narrow(0,
                                                      param_start,
                                                      numel_to_copy).copy_(part_to_copy)
            #param_offset += param.data.numel()
            param_offset += param.ds_tensor.numel()
            
            param.data = replicated_tensor.data

        return None

    def _reduce_scatter_gradients(self, param_list):
        #print_rank_0([param.grad for param in param_list])
        #assert any([param.grad is None for param in param_list]), "None gradients cannot be reduce scattered"

        handles_and_reduced_partitions = []
        for param in param_list:
            assert param.grad.numel() == param.ds_numel, f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params"

            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param))

        for param, (handle, reduced_partition) in zip(param_list, handles_and_reduced_partitions):
            if handle is not None:
                handle.wait()

            #some ranks may have partitions that are padded to go beyond the grad size.
            #For these ranks the output of reduce scatter is a separate buffer and needs
            #to be copied in
            partition_size = param.ds_tensor.numel()
            start = self.rank * partition_size
            end = start + partition_size
            #print_rank_0("REduce scatter was executed for praam {param.ds_id}")
            if start < param.ds_numel and end > param.ds_numel:
                elements = param.ds_numel - start
                param.grad.view(-1).narrow(0,
                                           start,
                                           elements).copy_(
                                               reduced_partition.narrow(0,
                                                                        0,
                                                                        elements))

    def _reduce_scatter_gradient(self, param):

        partition_size = param.ds_tensor.numel()
        #output = torch.empty(partition_size, dtype=param.dtype, device=param.device)

        total_size = partition_size * self.world_size
        input_list = []

        for i in range(self.world_size):

            start = i * partition_size
            end = start + partition_size

            #print("before reduce scatter gradients")
            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size)
            else:
                input = torch.zeros(partition_size,
                                    dtype=param.dtype,
                                    device=param.device)

                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0,
                                 0,
                                 elements).copy_(
                                     param.grad.view(-1).narrow(0,
                                                                start,
                                                                elements))
            #print("after reduce scatter gradients")
            input_list.append(input)

        rank = torch.distributed.get_rank(group=self.ds_process_group)
        handle = torch.distributed.reduce_scatter(input_list[rank],
                                                  input_list,
                                                  group=self.ds_process_group,
                                                  async_op=True)

        return handle, input_list[rank]

    def _partition_gradients(self, param_list, partition_buffers=None, accumulate=False):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list)

        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param, partition_buffer=partition_buffer, accumulate=accumulate)

    def _partition_gradient(self, param, partition_buffer=None, accumulate=False):
        #import pdb;pdb.set_trace()
        #param.grad=None
        #param.grad.test()
        print_rank_0(f"Partitioning gradient of size {param.grad.numel()}")
        see_memory_usage("Before partitioning gradients", force=False)
        partition_size = param.ds_tensor.numel()

        if partition_buffer is None:
            assert not accumulate, "No buffer to accumulate to"
            partition_buffer = torch.zeros(partition_size,
                                           dtype=param.dtype,
                                           device=param.device)
        else:
            assert partition_buffer.numel() >= partition_size, f"The partition buffer size {partition_buffer.numel()} should match the size of param.ds_tensor {partition_size}"

        rank = torch.distributed.get_rank(group=self.ds_process_group)
        start = partition_size * rank
        end = start + partition_size
        #print("before partition gradients")
        if start < param.ds_numel:
            elements = min(param.ds_numel - start, partition_size)

            dest_tensor = partition_buffer.view(-1).narrow(
                    0,
                    0,
                    elements)

            src_tensor = param.grad.view(-1).narrow(0,
                                                    start,
                                                    elements)

            #just copy the grad partition to the buffer
            if not accumulate:
                dest_tensor.copy_(src_tensor)

            #if source and destinatoin are on same device,
            #add to the provided buffer
            elif src_tensor.device == dest_tensor.device:
                dest_tensor.add_(src_tensor)

            #if source and destination are on different device, copy first to src
            #then add and move back to the destination. This seems to run faster
            #when src is gpu and dest is cpu
            #adding directly to cpu is very slow
            else:
                acc_tensor = torch.empty(src_tensor.numel(), dtype=param.dtype, device=param.device)
                acc_tensor.copy_(dest_tensor)
                acc_tensor.add_(src_tensor)
                dest_tensor.copy_(acc_tensor)


            # partition_buffer.view(-1).narrow(
            #     0,
            #     0,
            #     elements).copy_(param.grad.view(-1).narrow(0,
            #                                             start,
            #                                             elements))

        #print("after partition gradients")
        param.grad.data = dest_tensor.data
        see_memory_usage("After partitioning gradients", force=False)
