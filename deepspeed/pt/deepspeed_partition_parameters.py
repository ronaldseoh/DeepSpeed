import torch
from enum import Enum

def print_rank_0(message):
    if torch.distributed.get_rank() == 0:
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

#Inserts _post_init_method at the end of init method
#for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self):
        pass
        
    def __enter__(self):
        def partition_after(f):
            def wrapper(module, *args, **kwargs):
                print('beginning wrapped')
                f(module, *args, **kwargs)
                self._post_init_method(module)
                print('ending wrapped')
            
            return wrapper
    
        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)
        
        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _enable_class(subclass)

        #holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)


    def __exit__(self,exc_type, exc_value, traceback):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass

    #To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass


#Replaces all parameters in module with Scattered Parameters    
class ScatteredParameters(InsertPostInitMethodToModuleSubClasses):
    param_id = 0

    def __init__(self, ds_group=None):
        super(ScatteredParameters, self).__init__()
        assert torch.distributed.is_initialized(), "Parameters cannot be scattered without initializing torch.distributed"
        self.ds_process_group = torch.distributed.group.WORLD if ds_group is None else ds_group
        self.rank = torch.distributed.get_rank(group = self.ds_process_group)
        self.world_size = torch.distributed.get_world_size(group = self.ds_process_group)


    def _post_init_method(self, module):
        print(f'Converting Params in {module.__class__.__name__}' )
        for name, param in module.named_parameters(recurse=False):
            if not hasattr(param,'ds_id'):
                self._convert_to_deepspeed_param(param)
                print_rank_0(f"Partitioning param with ds id {param.ds_id} and shape {param.data.shape}")
                param.partition()


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

        #DeepSped Param ID
        param.ds_id = ScatteredParameters.param_id
        ScatteredParameters.param_id += 1

        def all_gather(param_list=None, async_op = False, hierarchy=None):
            cls = param
            if param_list is None:
                param_list=[cls]
            return self._all_gather(param_list, async_op = async_op, hierarchy=hierarchy)
            
        def partition(param_list=None, hierarchy=0):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Partitioning param with id {cls.ds_id}")
            if param_list is None:
                param_list = [cls]
            self._partition(param_list) 

        def reduce_gradients_to_owner(param_list=None, hierarchy=0):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Reducing Gradients for param with id {cls.ds_id} to owner")
            if param_list is None:
                param_list = [cls]
            self._reduce_scatter_gradients(param_list)

        def partition_gradients(param_list=None, partition_buffers=None):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Partitioning param gradient with id {cls.ds_id}")
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers,torch.Tensor):
                    partition_buffers = [partition_buffers]
            self._partition_gradients(param_list, partition_buffers = partition_buffers) 


        #Collectives for gathering and partitioning parameters
        param.all_gather = all_gather
        param.partition = partition
        
        #Collective for averaging gradients
        param.reduce_gradients_to_owner=reduce_gradients_to_owner
        param.partition_gradients = partition_gradients

        
    def _all_gather(self, param_list, async_op=False, hierarchy=None):
        handles = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                handle = self._allgather_param(param, async_op = async_op, hierarchy=hierarchy)
                param.ds_status = ZeroParamStatus.INFLIGHT if async_op else ZeroParamStatus.AVAILABLE
                handles.append(handle)
                    
        if async_op:
            return handles
        else:
            return None

    def _partition(self, param_list):
        for param in param_list:
            #print_rank_0(f"Partitioning Param {param.ds_id}")
            self._partition_param(param)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            if param.ds_tensor is not None:
                assert id(param.data) == id(param.ds_tensor.data), \
                "After the parameters are initially partitioned, make sure we are not recreating the partition."
           

    def _partition_param(self, param):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot parititon a param in flight"
        #print_rank_0(f"Param id {param.ds_id} status is {param.ds_status}")
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            if param.ds_tensor is not None:
                param.data = param.ds_tensor.data
                return
            
            tensor_size = param.ds_numel
            if tensor_size % self.world_size != 0:
                tensor_size += (self.world_size - (param.ds_numel % self.world_size))
            partition_size = tensor_size // self.world_size
            
            start = partition_size * self.rank
            end = start + partition_size

            one_dim_param = param.contiguous().view(-1)
            
            if start < param.ds_numel and end <= param.ds_numel:
                partitioned_tensor = one_dim_param.narrow(0,start, partition_size).clone().detach()
            else:
                partitioned_tensor = torch.zeros(partition_size, dtype = param.dtype, device = param.device)
                
                if start < param.ds_numel:
                    elements_to_copy = param.ds_numel - start
                    partitioned_tensor.narrow(0, 0, elements_to_copy).copy_(one_dim_param.narrow(0, start, elements_to_copy))

            param.ds_tensor = partitioned_tensor
            param.data = param.ds_tensor.data

            
            #print(f"ID {param.ds_id} partitioned and contains {param.data.shape}")

    def _allgather_param(self, param, async_op=False, hierarchy=0):
        
        partition_size = param.data.numel()
        tensor_size = partition_size * self.world_size
        flat_tensor = torch.zeros(param.ds_shape, dtype=param.dtype, device=param.device).view(-1)

        print_rank_0(f"{'--'* hierarchy}----Allgather param with id {param.ds_id} Partition Size {partition_size} and data shape {param.ds_shape}")
        partitions = []        
        for i in range(self.world_size):
        
            partitions.append(flat_tensor.narrow(0,partition_size * i, partition_size))
            
            if i == torch.distributed.get_rank(group= self.ds_process_group):
                partitions[i].copy_(param.data)
        
        handle = torch.distributed.all_gather(partitions, partitions[self.rank], group = self.ds_process_group, async_op = async_op)
        
        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        #param.data = flat_tensor.data
        #print(f"........ID {param.ds_id} gathered and contains {param.shape} and gradient {param.grad}")

        return handle

    def _reduce_scatter_gradients(self, param_list):
        assert any([para.grad is None for param in param_list]), "None gradients cannot be reduce scattered"
        assert any([param.grad.numel() !=  param.ds_numel for param in param_list]), "Cannot reduce scatter gradients whose size is not same as the params"
        
        handles_and_reduced_partitions = []
        for param in param_list:
            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param)
            
        for param, handle, reduced_partition in zip(param_list, handles_and_reduced_partitions):
            handle.wait()
            
            #some ranks may have partitions that are padded to go beyond the grad size. 
            #For these ranks the output of reduce scatter is a separate buffer and needs
            #to be copied in
            partition_size = param.ds_tensor.numel()
            start = i * partition_size
            end = start + partition_size

            if start < param.ds_numel and end > param.ds_numel:            
                param.grad.view(-1).narrow(0, start, elements).copy_(input_list[rank].narrow(0, 0, elements))


        
    def _reduce_scatter_gradient(self, param):

        partition_size = param.ds_tensor.numel()
        #output = torch.empty(partition_size, dtype=param.dtype, device=param.device)

        total_size = partition_size * self.world_size
        input_list = []

        for i in range(self.world_size):

            start = i * partition_size
            end = start + partition_size

            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size)
            else:
                input = torch.zeros(partition_size, dtype=param.dtype, device=param.device)
                
                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0, 0, elements).copy_(param.grad.view(-1).narrow(0, start, elements))
            
            input_list.append(input)
        
        rank = torch.distributed.get_rank(group=self.ds_process_group)
        handle = torch.distributed.reduce_scatter(input_list[rank],input_list, group = self.ds_process_group, async_op = True)
        
        return handle, input_list[rank]

    def _partition_gradients(self, param_list, partition_buffers = None):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list)

        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param, partition_buffer=partition_buffer)

    def _partition_gradient(self, param, partition_buffer=None):
        partition_size = param.ds_tensor.numel()
        
        if partition_buffer is None:
            partition_buffer = torch.zeros(partition_size, dtype=param.dtype, device=param.device)
        else:
            assert partition_buffer.numel() == partition_size, "The partition buffer size should match the size of param.ds_tensor"
        
        rank = torch.distributed.get_rank(group = self.ds_process_group)
        start = partition_size * rank
        end = start + partition_size

        if start < param.ds_numel:
            elements = min(param.ds_numel-start, partition_size)
            partition_buffer.view(-1).narrow(0, 0, elements).copy_(param.grad.view(-1).narrow(0, start, elements))
        
        param.grad.data=partition_buffer.data
