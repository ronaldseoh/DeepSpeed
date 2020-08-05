'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.distributed_c10d import _get_global_rank
import torch.distributed as dist
import math
from torch._six import inf
from torch.autograd import Variable

from deepspeed.pt.loss_scaler import LossScaler, DynamicLossScaler
from deepspeed.pt.deepspeed_utils import see_memory_usage, is_model_parallel_parameter
from deepspeed.pt.deepspeed_partition_parameters import ZeroParamStatus, ZeroParamType

import itertools
#Toggle this to true to enable correctness test
#with gradient partitioning and without
pg_correctness_test = False

from deepspeed.pt.log_utils import logger

try:
    from apex_C import flatten
    from apex_C import unflatten
except ImportError:
    try:
        _ = warned_flatten
    except NameError:
        logger.warning(
            "apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten."
        )
        warned_flatten = True
    from torch._utils import _flatten_dense_tensors as flatten
    from torch._utils import _unflatten_dense_tensors as unflatten

def print_rank_0(message, debug=True, force=False):
    if torch.distributed.get_rank() == 0 and (debug or force):
        print(message)

def input(msg):
    return


def split_half_float_double(tensors):
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor"
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def isclose(a, b, rtol=1e-09, atol=0.0):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def lcm(x, y):
    from fractions import gcd  # or can import gcd from `math` in Python 3
    return x * y // gcd(x, y)


#create a flat tensor aligned at the alignment boundary
def flatten_dense_tensors_aligned(tensor_list, alignment, pg):
    num_elements = 0
    for tensor in tensor_list:
        num_elements = num_elements + tensor.numel()

    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add,
                                 device=tensor_list[0].device,
                                 dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]

        num_elements = num_elements + elements_to_add
    else:
        padded_tensor_list = tensor_list

    return _flatten_dense_tensors(padded_tensor_list)


def move_to_cpu(param_list):
    for param in param_list:
        param.data = param.data.cpu()
        param.ds_tensor.data=param.data

def get_all_parameters(sub_module):
    return itertools.chain(sub_module.named_parameters(recurse=False), sub_module.ds_external_parameters())

#TODO Needs to be implemented
class PrefetchCoordinator(object):
    
    def __init__(self):
        #step_id keeps track of the number of sub-modules invoked so far
        #the step_id is tracking forward and backward sequence of sub-modules
        self.step_id = 0
        
        #stores the sequence of sub modules in forward+backward pass
        self.sub_module_trace=[]

        #maps sub_module id to submodule objects
        self.id_to_sub_module_map = {}

        #stores the total number of parmeters in each sub_module
        self.id_to_sub_module_size_map = {}

        self.trace_completed = False

        self.most_recent_sub_module_step = {}

        #reuse distances
        self.reuse_numel_for_step_id = {}
        
    def record_trace(self, sub_module):
        if not self.trace_completed:
            self.sub_module_trace.append(sub_module.id)
            self.id_to_sub_module_map[sub_module.id] = sub_module

    def print_trace(self):
        print_rank_0(f"The module trace is : {[self.id_to_sub_module_map[module_id].id for module_id in self.sub_module_trace]}")

    def increment_step(self, sub_module):
        self.most_recent_sub_module_step[sub_module.id] = self.step_id
        self.step_id += 1

    def reset_step(self):
        self.step_id = 0


    #returns the next numel parameters that will be used next but are not available or inflight 
    def get_params_to_prefetch(self, sub_module, numel=2000000):
        
        # numel_in_sub_module = 0
        # for name, param in sub_module.named_parameters(recurse=False):
        #     numel_in_sub_module += param.ds_numel

        # #if numel_in_sub_module < (numel // 2):
        #    return []

        #tracing failed. The sub_module passed at the step_id must match with the sub_module during tracing
        if sub_module.id != self.sub_module_trace[self.step_id]:
            print_rank_0(f"Tracing failed. Prefetching is disabled at sub-module: {sub_module.id}")
            return []

        params_to_prefetch = []
        total_numel_to_prefetch = 0
        
        for i in range(self.step_id, len(self.sub_module_trace)):
            module_id = self.sub_module_trace[i]
            for _, param in get_all_parameters(self.id_to_sub_module_map[module_id]):
                if param.ds_status is ZeroParamStatus.NOT_AVAILABLE and (param.ds_id not in [p.ds_id for p in params_to_prefetch]):
                    params_to_prefetch.append(param)
                    total_numel_to_prefetch += param.ds_numel
                    #print_rank_0(f"Total numel to prefetch: {total_numel_to_prefetch}. Param: {param.ds_shape} and numel {param.ds_numel}, numel limit {numel}")
                    if total_numel_to_prefetch >= numel: # and total_numel_to_prefetch > (numel_in_sub_module // 2):
                        return params_to_prefetch

        return params_to_prefetch

    #checks if this sub_module will be used again and if so then returns the number of elements
    #in the parameters used between this sub_module and the reuse of this sub_module
    def get_reuse_distance_in_numel(self, sub_module, sub_module_step_id = None):
        #assert is_forward is not None, "is_forward must be set to True for Forward Propagation and False for backward Propagation"
        is_there_reuse = False
        reuse_distance_in_numel = 1000000000000
        
        #set the appropriate trace
        trace = self.sub_module_trace
        total_steps = len(trace)
        if sub_module_step_id is None:
            sub_module_step_id = self.most_recent_sub_module_step[sub_module.id]

        #tracing failed. The sub_module passed at the step_id must match with the sub_module during tracing
        if sub_module.id != trace[sub_module_step_id]:
            print_rank_0(f"Tracing failed. Cannot tell if the sub_module: {sub_module.id} is reused")
            return reuse_distance_in_numel

        #return cached value
        if sub_module_step_id in self.reuse_numel_for_step_id:
            return self.reuse_numel_for_step_id[sub_module_step_id] 
            
        
        start_step = self.step_id
        print_rank_0(f"Step id is {self.step_id} ")
        for step_id in range(start_step, total_steps):
            print_rank_0(f"Trace id {trace[step_id]} and sub_module id {sub_module.id}")
            if sub_module.id == trace[step_id]:
                end_step = step_id
                
                is_there_reuse = True
                reuse_distance_in_numel = self._distance_in_numel(start_step, end_step, trace)        

                break

        self.reuse_numel_for_step_id[sub_module_step_id]  = reuse_distance_in_numel
        
        return reuse_distance_in_numel


    def _distance_in_numel(self, start_step, end_step, trace):
        distance_in_numel = 0
        for step_id in range(start_step, end_step):
            module_id = trace[step_id]
            for _, param in self.id_to_sub_module_map[module_id].named_parameters(recurse=False):
                distance_in_numel += param.ds_numel
            for _, param in self.id_to_sub_module_map[module_id].ds_external_parameters():
                distance_in_numel += param.ds_numel
        return distance_in_numel

class PartitionedParameterCoordinator(object):
    def __init__(self, comm_stream = None, max_reuse_distance_in_numel=500000000, max_available_parameters_in_numel=700000000):
        
        self.in_flight_handles = []
        self.params_in_flight = []
        self.comm_stream = comm_stream if comm_stream is not None else torch.cuda.current_stream()
        self.prefetch_coordinator = PrefetchCoordinator()
        self.hierarchy = 0

        self.total_available_parameter_numel = 0
        self.max_available_parameters_in_numel = max_available_parameters_in_numel

        #max distance between two use of the module beyond which module is released
        self.max_reuse_distance_in_numel = max_reuse_distance_in_numel

    def _increment_available_parameter_numel(self, increment):
        self.total_available_parameter_numel += increment

    def _decrement_available_parameter_numel(self, decrement):
        self.total_available_parameter_numel -= decrement

    '''-----------------------Tracing and Prefetching ---------------'''
    def record_trace(self, sub_module):
        self.prefetch_coordinator.record_trace(sub_module)

        
    def finish_tracing(self, print_trace=False):
        self.prefetch_coordinator.trace_completed = True
        
        if print_trace:
            self.prefetch_coordinator.print_trace()
    
    # Pre fetches the parameters for sub_modules that comes after 
    #  the current sub_module. This call is asynchronous
    def prefetch_next_sub_modules(self, sub_module, numel=5000000):

        params_to_prefetch = []
        if not self.prefetch_coordinator.trace_completed:
            return params_to_prefetch
        
        #prefetch if there is no current prefetching in flight
        if not self.in_flight_handles and self.total_available_parameter_numel < self.max_available_parameters_in_numel:
            params_to_prefetch = self.prefetch_coordinator.get_params_to_prefetch(sub_module, numel=numel)
            
            self._all_gather(params_to_prefetch, async_op = True)
            for param in params_to_prefetch:
                param.ds_status = ZeroParamStatus.INFLIGHT
                
                #keeping track of number of elements consumed by available parmaeters
                self._increment_available_parameter_numel(param.ds_numel)
        
        self._print_prefetch_elements_info(sub_module, params_to_prefetch)         
        print_rank_0(f"{'--' * self.hierarchy}--PreFetching parameters {[param.ds_id for param in params_to_prefetch]} and available {self.total_available_parameter_numel}, max limit {self.max_available_parameters_in_numel}", force = False)
    
    def _print_prefetch_elements_info(self, sub_module, params_to_prefetch):
        sub_module_numel = 0.0
        for name, param in sub_module.named_parameters(recurse=False):
            sub_module_numel += param.ds_numel
        numel_being_prefetched = 0
        for param in params_to_prefetch:
            numel_being_prefetched = param.ds_numel
        print_rank_0(f"{'--' * self.hierarchy}--PreFetching  {numel_being_prefetched} numels and number of numel in the next sub module is {sub_module_numel}", force=False)
    
    def increment_step(self, sub_module):
        self.prefetch_coordinator.increment_step(sub_module)

    def reset_step(self):
        self.prefetch_coordinator.reset_step()
    '''----------------------------------------------------------------------'''

    #Fetches the parameters in the sub_module
    #This call is blocking
    def fetch_sub_module(self, sub_module):
        partitioned_params = []
        params_in_flight = False
        #print_rank_0(f"{'--' * self.hierarchy}Fetching params in module {sub_module.__class__.__name__}")
        params_to_fetch = [param for _, param in sub_module.named_parameters(recurse=False)] 
        if hasattr(sub_module,'ds_external_parameters'):
            print_rank_0(f"{'--' * self.hierarchy}--Fetching external parameters {sub_module.ds_external_parameters()}")
            params_to_fetch += [param for _, param in sub_module.ds_external_parameters()]
        #for _, param in sub_module.named_parameters(recurse=False):
        for param in params_to_fetch: 
            param.ds_active_sub_modules += 1
            print_rank_0(f"{'--' * self.hierarchy}--Fetching parameters {param.ds_id} with active sub modules {param.ds_active_sub_modules}")
            
            if param.ds_status == ZeroParamStatus.AVAILABLE:
                print_rank_0(f"{'--' * self.hierarchy}--Parameter {param.ds_id} is already available")
                
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                print_rank_0(f"{'--' * self.hierarchy}--Parameter {param.ds_id} is being fetched")
                partitioned_params.append(param)

                #keeping track of number of elements consumed by available parmaeters
                self._increment_available_parameter_numel(param.ds_numel)
                print_rank_0(f"Incrementing with parameter id {param.ds_id}")

            if param.ds_status == ZeroParamStatus.INFLIGHT:
                params_in_flight = True
                print_rank_0(f"{'--' * self.hierarchy}--Parameters {param.ds_id} is already in flight (prefetched)")
        self.hierarchy += 1
            
        #parameters are partitioned and need to be allgathered
        self._all_gather(partitioned_params, async_op=True)
        
        #parameters are inflight and communication needs to be completed
        if partitioned_params or params_in_flight:
            self._synchronize_communication()

        for _, param in sub_module.named_parameters(recurse=False):
            param.ds_status = ZeroParamStatus.AVAILABLE

    def release_sub_module(self, sub_module):
        self.hierarchy -= 1
        print_rank_0(f"{'--' * self.hierarchy}Releasing params in module {sub_module.__class__.__name__}")
        params_to_release = [param for _, param in sub_module.named_parameters(recurse=False)] 
        if hasattr(sub_module, 'ds_external_parameters'):
            #print_rank_0(f"Releasing external parameters {sub_module.ds_external_parameters()}")
            params_to_release += [param for _, param in sub_module.ds_external_parameters()]
        
        #for _, param in sub_module.named_parameters(recurse=False):
        for param in params_to_release:
            param.ds_active_sub_modules -= 1
            if not param.ds_active_sub_modules and not self._keep_for_later(sub_module) and not param.ds_persist:
                print_rank_0(f"{'--' * self.hierarchy}--Releasing parameters {param.ds_id} with active sub modules {param.ds_active_sub_modules} and keep for later {self._keep_for_later(sub_module)}")
                
                #Keeping track of number of elements that are consumed by available parameters
                self._decrement_available_parameter_numel(param.ds_numel)
                param.partition(hierarchy=self.hierarchy)
                param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            else:
                print_rank_0(f"{'--' * self.hierarchy}--Did not release parameters {param.ds_id} with active sub modules {param.ds_active_sub_modules}, keep for later {self._keep_for_later(sub_module)} and persistence {param.ds_persist}")

    def release_and_reset_parameter(self,param):
        param.ds_active_sub_modules = 0
        if param.ds_status == ZeroParamStatus.AVAILABLE:
            print_rank_0(f"Releasing unpartitioned {param.ds_id} active sub-modules {param.ds_active_sub_modules} size {param.ds_numel} and persisitence {param.ds_persist}")
            self._decrement_available_parameter_numel(param.ds_numel)
            param.partition()
    
    def _keep_for_later(self, sub_module):
        if not self.prefetch_coordinator.trace_completed:
            return False
        reuse_distance_in_numel = self.prefetch_coordinator.get_reuse_distance_in_numel(sub_module)
        #print_rank_0(f"Reuse distance and numel for sub_module id {sub_module.id} is {reuse_distance_in_numel}")
        return reuse_distance_in_numel < self.max_reuse_distance_in_numel

    def _all_gather(self, partitioned_params, async_op = False):
        with torch.cuda.stream(self.comm_stream):
            handles = partitioned_params[0].all_gather(param_list = partitioned_params, async_op = async_op, hierarchy=self.hierarchy) if partitioned_params else None
        
        if handles is not None:
            self.in_flight_handles.extend(handles)
            self.params_in_flight.extend(partitioned_params)
        

    
    def _synchronize_communication(self, synchronize_streams=True):
        assert len(self.params_in_flight) == len(self.in_flight_handles)
        for handle, param in zip(self.in_flight_handles, self.params_in_flight):
            if handle is not None:
                with torch.cuda.stream(self.comm_stream):
                    handle.wait()
            param.ds_status = ZeroParamStatus.AVAILABLE
        self.comm_stream.synchronize()
        torch.cuda.synchronize() if synchronize_streams else None
        self.in_flight_handles = []
        self.params_in_flight = [] 


class PreBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args

class PostBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, extra_input,  *outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        ctx.extra_input = extra_input
        return outputs

    @staticmethod
    def backward(ctx, *args):    
        ctx.pre_backward_function(ctx.module)
        extra_input = ctx.extra_input
        #print_rank_0(f"Post backward function for module {ctx.module.__class__.__name__} and id {ctx.module.id}")
        if extra_input is not None:
            return (None, None) + (torch.ones(1, device=extra_input.device),) +  args 
        else:
            return (None, None, None) + args

        
class FP16_DeepSpeedZeroOptimizer_Stage3(object):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 mpu=None,
                 clip_grad=0.0,
                 allreduce_always_fp32=False,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0):

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Allgather bucket size {allgather_bucket_size}")
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master gard and unflat master weight never exist. TODO: a way to save out unflat master?
        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        self.module = module
        
        self.param_coordinator = PartitionedParameterCoordinator(comm_stream=torch.cuda.Stream())
        #self.param_coordinator = PartitionedParameterCoordinator()
        
        #-------------Stage 3 Setup-------------------#
        #parameters smaller than the threshold will be collectively gathered at the 
        #end of the optimizer step and will be kept till the end of the backward pass
        #TODO maybe worth just replicating these parameters and doing all reduce for them
        self.persistence_threshold = 100000
                
        self.persistent_parameters = self.persistent_parameters()
        
        self.setup_zero_stage3_hooks()
        
        
        #resetting ds_tensor just in case parameters have been changed after initialization
        #example .half() or .to()
        self.reset_ds_tensor()
        #---------------------------------------------#

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        self.overlap_comm = overlap_comm

        self.dp_process_group = dp_process_group

        self.partition_count = dist.get_world_size(group=self.dp_process_group)

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()

        
        self.overflow = False
        self.clip_grad = clip_grad
        self.allreduce_always_fp32 = allreduce_always_fp32
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients

        
        
        if self.reduce_scatter:
            assert not self.allreduce_always_fp32, "allreduce_always_fp32 is not yet supported with ZeRO-2 with reduce scatter enabled"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with ZeRO-2 with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with ZeRO-2 with reduce scatter enabled"

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []

        
        #a single 32-bit partition of the parallel partitioned parameters
        #that this process will update
        self.fp32_groups_flat = []

        
        #number of elements per partition in each group
        self.partition_size = []

        partition_id = dist.get_rank(group=self.dp_process_group)

        self.all_reduce_print = False

        self.prefetch_elements=25000000
        
        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])

            #not sure why apex was cloning the weights before flattening
            #removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")
            #move all the parameters to cpu to free up GPU space for creating flat buffer
            move_to_cpu(self.fp16_groups[i])
            see_memory_usage(f"After moving param group {i} to CPU")

            #create flat buffer in CPU and move to GPU
            self.fp16_groups_flat.append(
                flatten_dense_tensors_aligned(
                    self.fp16_groups[i],
                    dist.get_world_size(group=self.dp_process_group),
                    self.dp_process_group).cuda(torch.cuda.current_device()))
            see_memory_usage(f"After flattening and moving param group {i} to GPU")

            
            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
                p.ds_tensor.data = p.data

        
            # a partition of the fp32 master weights that will be updated by this process
            self.fp32_groups_flat.append(
                self.fp16_groups_flat[i].clone().float().detach())

            see_memory_usage(f"After creating fp32 copy {i}")

            # modify optimizer of have flat master weight
            self.fp32_groups_flat[i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.fp32_groups_flat[i]]

        
        see_memory_usage("Before initializing optimizer states")
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states")

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")

        
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)
        
        self.reduction_stream = torch.cuda.Stream() if self.overlap_comm else torch.cuda.current_stream()
        self.callback_queued = False

        self.param_dict = {}

        #map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.contiguous_gradients = contiguous_gradients
        self.extra_large_param_to_reduce = None
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None

        #simplified param id
        self.param_id = {}

        count = 0
        for i, params_group in enumerate(self.fp16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                count = count + 1

        
        #stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        
        #stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        
        #will store the averaged gradients required by this parititon
        self.averaged_gradients = {}

        
        #creates backward hooks for gradient partitioning
        self.create_reduce_and_remove_grad_hooks()

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

            self.dynamic_loss_scale = True

        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.cur_iter = 0

        
        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer")

    def reset_ds_tensor(self):
        for name, param in self.module.named_parameters(recurse=True):
            assert hasattr(param,'ds_id'), "Parameters have not been converted to be Zero 3 compatible"
            assert (param.ds_status == ZeroParamStatus.NOT_AVAILABLE), "All the parameters must have been partitioned by now"
            param.ds_tensor.data = param.data
    
    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0
        self._register_hooks_recursively(self.module)

    def persistent_parameters(self):
        persistent_params = []
        total_persistent_parameters = 0
        for _, param in self.module.named_parameters(recurse=True):
            if param.ds_numel < self.persistence_threshold:
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel

        print_rank_0(f'ZeRO 3: Total persistent parameters: {total_persistent_parameters}', force=True)
        return persistent_params

        
    def _register_hooks_recursively(self, module, count = [0]):
        my_count = count[0]
        module.id = my_count
        
        #print(f"{module.__class__} : {module.id}")
        
        for child in module.children():
            count[0] = count[0]+1
            self._register_hooks_recursively(child, count = count)

        def _pre_forward_module_hook(module, *args):
            self.pre_sub_module_forward_function(module)

        def _post_forward_module_hook(module, *args):
            self.post_sub_module_forward_function(module)

        def _pre_backward_module_hook(module, inputs, output):
        
            def _run_before_backward_function(sub_module):
                self.pre_sub_module_backward_function(sub_module)

            # identity autograd.function that executes _run_before_backward_function in backward
            return PreBackwardFunction.apply(module,_run_before_backward_function, output)

        def _post_backward_module_hook(module, inputs):
            
            def _run_after_backward_function(sub_module):
                self.post_sub_module_backward_function(sub_module)

            requires_grad = any([isinstance(v, torch.Tensor) and v.requires_grad for v in inputs])
            
            extra_input = None
            
            #creating a tensor requiring grad so that the packward pass for the PostBackwardFunction is called
            # if not requires_grad:
            #     print(f"Extra tensor with requires grad inserted in module {module.__class__.__name__} with id {module.id} ")
            #     sample_tensor = list(filter(lambda x: isinstance(x, torch.Tensor), inputs))[0]
            #     extra_input = torch.ones(1, dtype=torch.float, device=sample_tensor.device)  
            #     extra_input.requires_grad = True

            return PostBackwardFunction.apply(module,_run_after_backward_function, extra_input, *inputs)


        #Pre forward hook
        module.register_forward_pre_hook(_pre_forward_module_hook)
        #Post forward hook
        module.register_forward_hook(_post_forward_module_hook)
        
        #Pre backward hook
        module.register_forward_hook(_pre_backward_module_hook)    

        # post backward hook
        module.register_forward_pre_hook(_post_backward_module_hook)
        

    def pre_sub_module_forward_function(self, sub_module):
        
        self.param_coordinator.record_trace(sub_module)
        
        self.param_coordinator.fetch_sub_module(sub_module)
        
        self.param_coordinator.prefetch_next_sub_modules(sub_module, numel=self.prefetch_elements)
        
        self.param_coordinator.increment_step(sub_module)


    def post_sub_module_forward_function(self, sub_module):
        self.param_coordinator.release_sub_module(sub_module)
    
    def pre_sub_module_backward_function(self, sub_module):
        self.param_coordinator.record_trace(sub_module)

        self.param_coordinator.fetch_sub_module(sub_module)
        
        self.param_coordinator.prefetch_next_sub_modules(sub_module, numel=self.prefetch_elements)

        self.param_coordinator.increment_step(sub_module)

    def post_sub_module_backward_function(self, sub_module):
            self.param_coordinator.release_sub_module(sub_module)


    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            self.grads_in_partition = None
            self.grads_in_partition_offset = 0

    def initialize_optimizer_states(self):

        for i, group in enumerate(self.fp16_groups):
            single_grad_partition = torch.zeros(
                int(self.fp32_groups_flat[i].numel()),
                dtype=self.fp32_groups_flat[i].dtype,
                device=torch.cuda.current_device())
            self.fp32_groups_flat[i].grad = single_grad_partition

        self.optimizer.step()

        for group in self.fp32_groups_flat:
            group.grad = None

        return

    #########################################################################
    #########################ZeRO Partition Gradients########################
    #########################################################################

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def initialize_gradient_partitioning_data_structures(self):

        total_partitions = dist.get_world_size(group=self.dp_process_group)

        for i, param_group in enumerate(self.fp16_groups):

            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}

            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][
                    partition_id] = self.get_first_param_index(
                        i,
                        param_group,
                        partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        if self.overlap_comm:
            self.reduction_stream.synchronize()

        self.partition_previous_reduced_grads()

        #if dist.get_rank() == 0:
        #    logger.info("Params already reduced %s", self.params_already_reduced)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        
        for i, _ in enumerate(self.fp16_groups):
            self.averaged_gradients[i] = self.get_flat_partition(
                self.fp16_groups[i],
                0,
                self.fp32_groups_flat[i].numel(),
                return_tensor_list=True)

        self._release_ipg_buffers()

        see_memory_usage(f"End ipg_epilogue")

    # resets all partition to no reduced
    # sets remianing grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    def reset_partition_gradient_structures(self):
        total_partitions = dist.get_world_size(group=self.dp_process_group)
        for i, _ in enumerate(self.fp16_groups):
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][
                    partition_id] = self.total_grads_in_partition[i][partition_id]

                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def initialize_gradient_partition(self, i, param_group, partition_id):
        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1

        partition_size = self.partition_size[i]

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for param in param_group:

            param_size = param.numel()
            param_id = self.get_param_id(param)

            if (current_index >= start_index and current_index < end_index):
                set_key_value_list(self.param_to_partition_ids[i],
                                   param_id,
                                   partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][
                    param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif start_index > current_index and start_index < (current_index +
                                                                param_size):
                assert (first_offset==0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i],
                                   param_id,
                                   partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def create_reduce_and_remove_grad_hooks(self):
        self.grad_accs = []
        for i, param_group in enumerate(self.fp16_groups):
            for param in param_group:
                if param.requires_grad:

                    #The hook must be created in un-partitioned parameter
                    param.all_gather()
                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)

                    wrapper(param, i)
                    
                    #Partition the parameter after creating the hook
                    param.partition()
        #exit(0)

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
        see_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}"
        )

    ###############Idependent Partition Gradient ########################
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        #print_rank_0(f"Inside reduce ipg buckets. Param ID {param.ds_id}, ipg elements {self.elements_in_ipg_bucket}, reduce bucket size {self.reduce_bucket_size}", force=True)
        if self.elements_in_ipg_bucket + param.ds_numel > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads",
                                         param.ds_numel)
            
            self.reduce_ipg_grads()

            
            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.ipg_index = 1 - self.ipg_index
            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads",
                                         param.ds_numel)

        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        
        #keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
        if param.ds_numel > self.reduce_bucket_size:
            self.extra_large_param_to_reduce = param

        elif self.contiguous_gradients:
            #print_rank_0("before new grad tensor move")
            new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(
                0,
                self.elements_in_ipg_bucket,
                param.ds_numel)
            #print_rank_0("after new grad tensor move")
            new_grad_tensor.copy_(param.grad.view(-1))
            param.grad.data = new_grad_tensor.data.view_as(param.grad)

        self.elements_in_ipg_bucket += param.ds_numel
        self.grads_in_ipg_bucket.append(param.grad)
        self.params_in_ipg_bucket.append((i, param, param_id))
        self.report_ipg_memory_usage("End ipg_remove_grads", 0)

    def gradient_reduction_w_predivide(self, tensor):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        tensor_to_allreduce = tensor

        if self.allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()

        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

            if self.gradient_predivide_factor() != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor() /
                                         dp_world_size)
        else:
            tensor_to_allreduce.div_(dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

        if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def average_tensor(self, tensors, params_to_reduce):
        with torch.cuda.stream(self.reduction_stream):
            if not self.reduce_scatter:
                for tensor in tensors:
                    self.gradient_reduction_w_predivide(tensor)
                return

            for tensor in tensors:
                tensor.div_(dist.get_world_size(group=self.dp_process_group))

            #reduction resulting with each rank only holding the gradient partition it owns
            #This could either be a reduce scatter or a reduce op depending on how 
            #parameters are partitionied. The method is impelemnted by the 
            #DeepSpeed param extensions to the pytroch parameter, so its up to
            #the extension to define what happens here
            params_to_reduce[0].reduce_gradients_at_owner(param_list=params_to_reduce, hierarchy=self.param_coordinator.hierarchy)


    def partition_previous_reduced_grads(self):
        if not self.previous_reduced_grads:
            return
 
        if self.contiguous_gradients and self.grads_in_partition is None:
            self.grads_in_partition_offset = 0
            total_size = 0
            for group in self.fp16_groups:
                for param_in_partition in group:
                    total_size += param_in_partition.ds_tensor.numel()

            see_memory_usage(f"before copying {total_size} gradients into partition")
            self.grads_in_partition = torch.empty(int(total_size),
                                                  dtype=torch.half,
                                                  device=torch.cuda.current_device())
            see_memory_usage(f"after copying {total_size} gradients into partition")

        
        for param in self.previous_reduced_grads:

            if not self.contiguous_gradients:
                param.partition_gradients(partition_buffer=None)
                continue

            #The allreduce buffer will be rewritted. Copy the gradients in partition to a new buffer
            new_grad_tensor = self.grads_in_partition.narrow(0,
                                                        self.grads_in_partition_offset,
                                                        param.ds_tensor.numel())
       
            param.partition_gradients(partition_buffers=new_grad_tensor)

            self.grads_in_partition_offset += param.ds_tensor.numel()
            
        self.previous_reduced_grads=[]           

    def reduce_ipg_grads(self, extra_param=None):
        if self.overlap_comm:
            self.reduction_stream.synchronize()
        self.partition_previous_reduced_grads()
                      
        params_to_reduce = [param for i, param, param_id in self.params_in_ipg_bucket]
        #print_rank_0(f"Reducing {[(param.ds_id, param.grad) for param in params_to_reduce]}")
        if self.contiguous_gradients:
            reduction_list = [self.ipg_buffer[self.ipg_index]]
            if self.extra_large_param_to_reduce is not None:
                reduction_list.append(self.extra_large_param_to_reduce.grad)
                self.extra_large_param_to_reduce=None
            self.average_tensor(reduction_list, params_to_reduce)
        else:
            self.buffered_reduce_fallback(
                None,
                self.grads_in_ipg_bucket,
                elements_per_buffer=self.elements_in_ipg_bucket)

        for _, param, param_id in self.params_in_ipg_bucket:
            self.params_already_reduced[param_id] = True
        
        self.previous_reduced_grads = params_to_reduce
                                        
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        #####################################################################

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):
        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True

        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None

    def flatten_and_print(self, message, tensors, start=0, n=5):
        flatten_tensor = _flatten_dense_tensors(tensors)

        def print_func():
            logger.info(flatten_tensor.contiguous().view(-1).narrow(0, start, n))

        self.sequential_execution(print_func, message)

    def get_grads_to_reduce(self, i, partition_id):
        def get_reducable_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(
                total_elements - start,
                self.partition_size[i] -
                self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0,
                                                             int(start),
                                                             int(num_elements))
            else:
                if num_elements == total_elements:
                    return grad.clone()
                else:
                    return grad.clone().contiguous().view(-1).narrow(
                        0,
                        int(start),
                        int(num_elements))

        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducable_portion(key)
            grads_to_reduce.append(grad)
        return grads_to_reduce

    def sequential_execution(self, function, message, group=None):
        if group is None:
            group = self.dp_process_group
        if dist.get_rank(group=group) == 0:
            logger.info(message)
        for id in range(dist.get_world_size(group=group)):
            if id == dist.get_rank(group=group):
                function()
            dist.barrier(group=group)

    def set_none_gradients_to_zero(self, i, partition_id):
        for param_id in self.is_grad_computed[i][partition_id]:
            param = self.param_dict[param_id]
            if param.grad is None:
                param.grad = torch.zero_like(param)

    ######################Reduction Related Methods##############################

    def allreduce_bucket(self, bucket, allreduce_always_fp32=False, rank=None, log=None):
        rank = None
        tensor = flatten(bucket)

        tensor_to_allreduce = tensor

        if pg_correctness_test:
            allreduce_always_fp32 = True

        if allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()

        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            global_rank = _get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

        if allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    #if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        with torch.cuda.stream(self.reduction_stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self,
                            bucket,
                            numel_per_bucket=500000000,
                            rank=None,
                            log=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    #allows using reduction of gradients instead of using all_reduce
    def buffered_reduce_fallback(self,
                                 rank,
                                 grads,
                                 elements_per_buffer=500000000,
                                 log=None):
        split_buckets = split_half_float_double(grads)

        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket,
                                     numel_per_bucket=elements_per_buffer,
                                     rank=rank,
                                     log=log)

    #############################################################################
    #############################################################################
    #############################################################################

    #views the tensor as multiple partitions and returns
    #those partitions
    def get_data_parallel_partitions(self, tensor):
        partitions = []

        dp = dist.get_world_size(group=self.dp_process_group)
        dp_id = dist.get_rank(group=self.dp_process_group)

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if (current_index >= start_index and current_index < end_index):
                params_in_partition.append(tensor)

            elif start_index > current_index and start_index < (current_index +
                                                                tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset==0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None:
            torch.distributed.all_reduce(tensor=tensor, op=op)
        else:
            torch.distributed.all_reduce(tensor=tensor,
                                         op=op,
                                         group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                            op=torch.distributed.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            #if dist.get_rank() == 0:
            #    logger.info(f"Total Norm begining {total_norm}")
            for g, p in zip(gradients, params):
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item()**2
            # Sum across all model parallel GPUs.
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                            op=torch.distributed.ReduceOp.SUM)

            total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float(
                'inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    #creates a flat fused tensor from the tensor list starting at the first_offset
    #in the first tensor of the list. If there are not enough elements in the tensor
    #list then the flat tensor will be padded with zeros
    def get_flat_partition(self,
                           tensor_list,
                           first_offset,
                           partition_size,
                           return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0
        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                print_rank_0(f"Warning some of the gradients are None")
                continue

            tensor = tensor.grad
            num_elements = tensor.numel()
            tensor_offset = 0

            #we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            #we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size

            #we need a narrow view of the tensor based on the tensor offset and number of elements that
            #we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                    0,
                    int(tensor_offset),
                    int(num_elements)))
            else:
                flat_tensor_list.append(tensor)

            current_size = current_size + num_elements

        #this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            flat_tensor_list.append(
                torch.zeros(int(partition_size - current_size),
                            dtype=tensor_list[0].dtype,
                            device=tensor_list[0].device))

        if return_tensor_list:
            return flat_tensor_list

        return _flatten_dense_tensors(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None

    def step(self, closure=None):
        """
        Not supporting closure.
        """

        print_rank_0(f"Inside Step function")
        see_memory_usage(f"In step before checking overflow")

        print_rank_0("Finished Tracing at Beginning of Step")
        self.param_coordinator.hierarchy = 0
        self.param_coordinator.finish_tracing(print_trace=True)

        self.param_coordinator.reset_step()

        print_rank_0("Finished Tracing at Beginning of Step")

        
        # First compute norm for all group so we know if there is overflow
        self.check_overflow()

        timers = self.timers

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad()
            see_memory_usage('After overflow after clearing gradients')

            logger.info(
                "[deepscale] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, "
                "reducing to {}".format(dist.get_rank(),
                                        prev_scale,
                                        self.loss_scale))
            timers('optimizer_step').start()
            timers('optimizer_step').stop()
            timers('optimizer_allgather').start()
            timers('optimizer_allgather').stop()
            return

        norm_groups = []
        single_partition_grad_groups = []
        skip = False
        partition_id = dist.get_rank(group=self.dp_process_group)
        for i, group in enumerate(self.fp16_groups):

            norm_groups.append(
                self.get_grad_norm_direct(self.averaged_gradients[i],
                                          self.fp16_groups[i]))

            #free gradients for all the prameters that are not updated by this process
            #self.free_grad_in_param_list(self.params_not_in_partition[i])

            #create a flat gradients for parameters updated by this process
            # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
            single_grad_partition = _flatten_dense_tensors(
                        self.averaged_gradients[i]).to(
                        self.fp32_groups_flat[i].dtype)
            assert single_grad_partition.numel() == self.fp32_groups_flat[i].numel(), \
                "averaged gradients have different number of elements that partition size {} {} {} {}".format(single_grad_partition.numel(), self.partition_size[i], i, partition_id)

            self.fp32_groups_flat[i].grad = single_grad_partition
            
            
            #release all the gradient since we have already created a necessary copy in dp_grad_partition
            self.zero_grad()

            self.averaged_gradients[i] = None

            single_partition_grad_groups.append(single_grad_partition)

        self.unscale_and_clip_grads(single_partition_grad_groups, norm_groups)

        timers('optimizer_step').start()
        self.optimizer.step()
        #get rid of the fp32 gradients. Not needed anymore
        for group in self.fp32_groups_flat:
            group.grad = None


        for fp16_partitions, fp32_partition in zip(self.fp16_groups_flat, self.fp32_groups_flat):
            fp16_partitions.data.copy_(fp32_partition.data)
        timers('optimizer_step').stop()

        
        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
                p.ds_tensor.data = q.data

        #Gathering persisting parameters
        self.persistent_parameters[0].all_gather(self.persistent_parameters)

        see_memory_usage('After zero_optimizer step')
        print_rank_0(f"------------------Finishing Step-----------------------")
        return

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm**2.0
        total_norm = math.sqrt(total_norm)

        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.fp16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.has_overflow_partitioned_grads_serial()
            overflow_gpu = torch.cuda.ByteTensor([overflow])
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

        else:
            params = []
            for group in self.fp16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = torch.cuda.ByteTensor([overflow])

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu,
                                        op=torch.distributed.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(self.reduce_bucket_size,
                                dtype=torch.half,
                                device=torch.cuda.current_device())
            self.ipg_buffer.append(buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(self.reduce_bucket_size,
                                    dtype=torch.half,
                                    device=torch.cuda.current_device())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0

        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)
        
        '''Partitioning Parameters that were not partitioned 
        Usually if parameters of modules whose input parameters do not require
        grad computation do not trigger post call and will therefore will remain unpartitioned'''
        for name, param in self.module.named_parameters(recurse=True):
            self.param_coordinator.release_and_reset_parameter(param)
            
    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict[
            'single_partition_of_fp32_groups'] = self.single_partition_of_fp32_groups

        state_dict['partition_count'] = self.partition_count

        return state_dict

    # Refresh the fp32 master params from the fp16 copies.
    def refresh_fp32_params(self):
        partition_id = dist.get_rank(group=self.dp_process_group)
        for fp16_partitions, fp32_partition in zip(self.parallel_partitioned_fp16_groups, self.single_partition_of_fp32_groups):
            fp32_partition.data.copy_(fp16_partitions[partition_id].data)

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']

        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 1 if changing DP degree and option 2 otherwise.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.

        if 'partition_count' in state_dict and state_dict[
                'partition_count'] == self.partition_count:
            # Use option 2
            for current, saved in zip(self.single_partition_of_fp32_groups, state_dict['single_partition_of_fp32_groups']):
                current.data.copy_(saved.data)
        else:
            # Use option 1
            partition_id = dist.get_rank(group=self.dp_process_group)
            for fp16_partitions, fp32_partition in zip(self.parallel_partitioned_fp16_groups, self.single_partition_of_fp32_groups):
                fp32_partition.data.copy_(fp16_partitions[partition_id].data)


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = torch.distributed.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        logger.info(
            f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}"
        )
