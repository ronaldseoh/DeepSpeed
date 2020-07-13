import torch
from enum import enum

class ZeroParamType(enum):
    
    #same as regular pytorch parameters
    NORMAL = 1
    
    #parameters are partitioned across data parallel process
    PARTITIONED = 2
    
    #the parameter is held with a unique process rank 
    #and is not available on all other process
    REMOTE = 3

class ZeroParamStatus(enum):
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
        
    def __enter__(self):
        def partition_after(f):
            @wraps(f)
            def wrapper(module, *args, **kwargs):
                print('beginning wrapped')
                f(module, *args, **kwargs)
                self._post_init_function(module)
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


    def __exit__(self):
        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = _old_init_subclass

    #To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass


#Replaces all parameters in module with Scattered Parameters    
class ScatterdParameters(InsertPostInitMethodToModuleSubClasses):
    def __init__(ds_group=None):
        super(ScatterParameters, self).__init__()
        assert torch.distributed.is_initialized(), "Parameters cannot be scattered without initializing torch.distributed"
        self.ds_process_group = torch.distributed.group.WORLD if ds_group is None else ds_group
        self.rank = torch.distributed.get_rank(group = self.ds_process_group)
        self.world_size = torch.distributed.get_world_size(group = self.ds_process_group)


    def _post_init_method(self, module):
        print(f'SCATTERING PARAMS in {module.__class__.__name__}' )
        for name, param in module.named_parameters(recurse=False):
            self._convert_to_deepspeed_param(param, name)
            param.partition()


    def _convert_to_deepspeed_param(self, param):
        
        # Partitioned, Normal, Remote 
        param.ds_param_type = ZeroParamType.PARTITIONED
        
        # Replicated vs Partitioned vs Inflight
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

        #Stores the shape of the original tensor
        param.ds_shape = list(param.shape.numpy())

        #Stores the number of elements in the original parmaeter without padding
        param.ds_numel = param.numel()

        #Stores the paritioned copy of the tensor
        param.ds_tensor = None

        def allgather(cls, param_list=None, async_op = False):
            if param_list is None:
                param_list=[cls]
            return self._allgather(param_list, async_op = async_op)
            
        def partition(cls, param_list=None):
            if param_list is None:
                param_list = [cls]
            self._partition(param_list) 

        param.allgather = allgather
        param.partition = partition


    def _allgather(self, param_list, async_op=False):
        handles = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                handle = self._param_allgather(param, async_op = async_op)
                param.ds_status = ZeroParamStatus.INFLIGHT if async_op else ZeroParamStatus.AVAILABLE
                handles.append(handle)
                    
        
        return handles if async_op else return None

    def _partition(self, param_list):
        for param in param_list:
            self._partition_param(param)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
           

    def _partition_param(self, param):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot parititon a param in flight"
        
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            if param.ds_tensor is not None:
                partitioned_tensor = param.ds_tensor   
            else:
                tensor_size = param.numel() + (self.world_size - (param.numel() % self.world_size))
                partition_size = tensor_size // self.world_size
                
                start = partition_size * self.rank
                end = start + partition_size

                one_dim_param = param.contiguous().view(-1)
                
                if start < param.numel() and end <= param.numel():
                    partitioned_tensor = one_dim_param.narrow(0,start, partition_size).clone().detach()
                else:
                    partitioned_tensor = torch.zeros(partition_size, dtype = param.dtype, device = param.device)
                    
                    if start < param.numel():
                        elements_to_copy = param.numel() - start
                        partitioned_tensor.narrow(0, 0, elements_to_copy).copy_(one_dim_param.narrow(0, start, elements_to_copy))

            param.data = partitioned_tensor.data

    def _allgather_param(self, param, async_op=False):
        #Storing partitioned copy
        param.ds_tensor = param.data
        
        partition_size = param.data.numel()
        tensor_size = partition_size * self.world_size
        flat_tensor = torch.zeros([tensor_size], dtype=param.dtype, device=param.device)

        partitions = []        
        for i in range(self.world_size):
        
            partitions.append(flat_tensor.narrow(0,partition_size * i, partition_size))
            partitions[i].copy_(param.data)
        
        handle = dist.all_gather(partitions, partitions[self.rank], group = self.ds_process_group, async_op = async_op)
        
        replicated_tensor = flat_tensor.narrow(0, 0, ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        return handle



