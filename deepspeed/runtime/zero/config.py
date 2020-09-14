"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.utils import logger
from deepspeed.runtime.zero.constants import *


class DeepSpeedZeroConfig(object):
    def __init__(self, param_dict):
        super(DeepSpeedZeroConfig, self).__init__()

        self.stage = None
        self.contiguous_gradients = None
        self.reduce_scatter = None
        self.reduce_bucket_size = None
        self.allgather_partitions = None
        self.allgather_bucket_size = None
        self.overlap_comm = None
        self.load_from_fp32_weights = None
        self.cpu_offload = None

        #Stage3 Specific Parameters
        self.prefetch_bucket_size = None
        self.param_persistence_threshold = None
        self.max_live_parameters = None
        self.max_reuse_distance = None

        #Stage3 Specific Parameters
        self.prefetch_bucket_size = None
        self.param_persistence_threshold = None
        self.max_live_parameters = None
        self.max_reuse_distance = None

        if ZERO_OPTIMIZATION in param_dict.keys():
            zero_config_dict = param_dict[ZERO_OPTIMIZATION]
            if type(zero_config_dict) is bool:
                zero_config_dict = self.read_zero_config_deprecated(param_dict)
        else:
            zero_config_dict = ZERO_OPTIMIZATION_DEFAULT

        self._initialize(zero_config_dict)

    def read_zero_config_deprecated(self, param_dict):
        zero_config_dict = {}
        zero_config_dict[
            ZERO_OPTIMIZATION_STAGE] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
        if zero_config_dict[ZERO_OPTIMIZATION_STAGE] > 0:
            zero_config_dict[ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE] = get_scalar_param(
                param_dict,
                ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED,
                ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)

        logger.warning(
            'DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}'
            .format(ZERO_FORMAT))
        return zero_config_dict

    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def _initialize(self, zero_config_dict):
        self.stage = get_scalar_param(zero_config_dict,
                                      ZERO_OPTIMIZATION_STAGE,
                                      ZERO_OPTIMIZATION_STAGE_DEFAULT)

        self.contiguous_gradients = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS,
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT)

        self.reduce_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT)

        self.reduce_scatter = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)

        self.overlap_comm = get_scalar_param(zero_config_dict,
                                             ZERO_OPTIMIZATION_OVERLAP_COMM,
                                             ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT)

        self.allgather_partitions = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT)

        self.allgather_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)

<<<<<<< HEAD:deepspeed/runtime/zero/config.py
<<<<<<< HEAD:deepspeed/runtime/zero/config.py
        self.load_from_fp32_weights = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS,
            ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT)

        self.cpu_offload = get_scalar_param(zero_config_dict,
                                            ZERO_OPTIMIZATION_CPU_OFFLOAD,
                                            ZERO_OPTIMIZATION_CPU_OFFLOAD_DEFAULT)
=======
=======
>>>>>>> 8aca9e8ade56a7cf913a2ec802d4dab5430cb9f5:deepspeed/pt/deepspeed_zero_config.py
        self.max_live_parameters = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS,
            ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS_DEFAULT)

        self.max_reuse_distance = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE,
            ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE_DEFAULT)

        self.prefetch_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE,
            ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE_DEFAULT)

        self.param_persistence_threshold = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD,
            ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD_DEFAULT)


<<<<<<< HEAD:deepspeed/runtime/zero/config.py
>>>>>>> 93b6fdd... Added DeepSpeed Linear with save_for_backward tensor_id support. Added support to change zero-3 features from json. support for using pre-allocated buffer to avoid memory fragmentation (brittle prototype).:deepspeed/pt/deepspeed_zero_config.py
=======
>>>>>>> 8aca9e8ade56a7cf913a2ec802d4dab5430cb9f5:deepspeed/pt/deepspeed_zero_config.py
