"""
Copyright 2020 The Microsoft DeepSpeed Team

Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import torch
from deepspeed.runtime.constants import AIO_BLOCK_SIZE, AIO_QUEUE_DEPTH, \
    AIO_THREAD_COUNT, AIO_SINGLE_SUBMIT, AIO_OVERLAP_EVENTS
from deepspeed.runtime.constants import SWAP_DEEPSPEED_COPY
from deepspeed.ops.aio import aio_handle, deepspeed_memcpy
from deepspeed.utils.logging import logger


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        swap_handle.async_pread(buffer, path)

def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        swap_handle.async_pwrite(buffer, path)

class OptimizerSwapTensor(object):
    def __init__(self, swap_config, optimizer, base_folder, timers):
        self.swap_config = swap_config
        self.swap_tensors = self._get_state_tensors(optimizer)
        self.swap_buffers = self._create_swap_buffers(self.swap_tensors)
        self.swap_folder = base_folder
        os.makedirs(self.swap_folder, exist_ok=True)
        self.swap_paths = self._create_swap_paths(self.swap_tensors)
        self.timers = timers

    def release_memory(self):
        for tensor_list in self.swap_tensors.values():
            for tensor in tensor_list:
                tensor.data = torch.Tensor()

    def _get_state_tensors(self, optimizer):
        tensor_dict = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                tensor_list = []
                state = optimizer.state[p]
                for key in state.keys():
                    if torch.is_tensor(state[key]):
                        tensor_list.append(state[key])
                tensor_dict[p] = tensor_list

        return tensor_dict

    def _create_swap_paths(self, swap_tensors):
        swap_paths = {}
        for p in swap_tensors.keys():
            path_list = [os.path.join(self.swap_folder, f'{id(t)}.tensor.swp') for t in swap_tensors[p]]
            swap_paths[p] = path_list

        return swap_paths


    def _get_swap_buffers(self, param):
        if param.size() < self.swap_buffers[0].size():
            swap_buffers = [buffer[:len(param)] for buffer in self.swap_buffers]
        else:
            swap_buffers = self.swap_buffers
        return swap_buffers

    def swap_in(self, aio_handle):
        ALLOC_TIMER = 'swap_allocate'
        READ_TIMER = 'swap_submit_read'
        WAIT_TIMER = 'swap_wait_read'
        COPY_TIMER = 'swap_copy_read'

        copy_bytes = 0
        for p in self.swap_tensors.keys():
            self.start_timer(ALLOC_TIMER)
            swap_buffers = self._get_swap_buffers(p)
            self.stop_timer(ALLOC_TIMER)

            self.start_timer(READ_TIMER)
            swap_in_tensors(
                aio_handle,
                swap_buffers,
                self.swap_paths[p])
            self.stop_timer(READ_TIMER)

            self.start_timer(WAIT_TIMER)
            aio_handle.wait()
            self.stop_timer(WAIT_TIMER)

            self.start_timer(COPY_TIMER)
            for dst, src in zip(self.swap_tensors[p], swap_buffers):
                dst.data = torch.zeros(p.size(), dtype=p.dtype, device=p.device)
                if dst.device.type == 'cpu' and self.swap_config[SWAP_DEEPSPEED_COPY]:
                    deepspeed_memcpy(dst.data, src.data)
                else:
                    dst.data.copy_(src.data)
                copy_bytes += dst.numel() * dst.element_size()
            self.stop_timer(COPY_TIMER)

        self.log_timers([ALLOC_TIMER, READ_TIMER, WAIT_TIMER, COPY_TIMER])
        if torch.distributed.get_rank() == 0:
            logger.info(f'tensor_swap_in: {(copy_bytes/(1024**3)):5.2f} GB')


    def swap_out(self, aio_handle):
        ALLOC_TIMER = 'swap_allocate'
        WRITE_TIMER = 'swap_submit_write'
        WAIT_TIMER = 'swap_wait_write'
        COPY_TIMER = 'swap_copy_write'

        copy_bytes = 0
        for p in self.swap_tensors.keys():
            self.start_timer(ALLOC_TIMER)
            swap_buffers = self._get_swap_buffers(p)
            self.stop_timer(ALLOC_TIMER)

            self.start_timer(COPY_TIMER)
            for dst, src in zip(swap_buffers, self.swap_tensors[p]):
                if dst.device.type == 'cpu' and self.swap_config[SWAP_DEEPSPEED_COPY]:
                    deepspeed_memcpy(dst.data, src.data)
                else:
                    dst.data.copy_(src.data)
                src.data = torch.Tensor()
                copy_bytes += dst.numel() * dst.element_size()
            self.stop_timer(COPY_TIMER)

            self.start_timer(WRITE_TIMER)
            swap_out_tensors(
                    aio_handle,
                    swap_buffers,
                    self.swap_paths[p]
                )
            self.stop_timer(WRITE_TIMER)

            self.start_timer(WAIT_TIMER)
            aio_handle.wait()
            self.stop_timer(WAIT_TIMER)

        self.log_timers([ALLOC_TIMER, WRITE_TIMER, WAIT_TIMER, COPY_TIMER])
        if torch.distributed.get_rank() == 0:
            logger.info(f'tensor_swap_out: {(copy_bytes/(1024**3)):5.2f} GB')

    def _get_tensor_structure(self, tensor_dict):
        tensor_sizes = []
        dtype = None
        device = None
        for param in tensor_dict.keys():
            if len(tensor_sizes) == 0:
                dtype = param.dtype
                device = param.device

            tmp_sizes = [t.size() for t in tensor_dict[param]]
            if len(tmp_sizes) > len(tensor_sizes):
                padding = [torch.Tensor().size()] * (len(tmp_sizes) - len(tensor_sizes))
                tensor_sizes.extend(padding)
            tensor_sizes = [max(a, b) for a, b in zip(tensor_sizes, tmp_sizes)]

        return tensor_sizes, dtype, device

    def _create_swap_buffers(self, swap_tensors):
        sizes, dtype, device = self._get_tensor_structure(swap_tensors)
        swap_buffers = []
        for size in sizes:
            if device.type == 'cpu':
                buffer = torch.zeros(size, device=device, dtype=dtype).pin_memory()
            else:
                buffer = torch.zeros(size, device=device, dtype=dtype)
            swap_buffers.append(buffer)
        return swap_buffers

    def start_timer(self, name):
        if self.timers:
            self.timers(name).start()

    def stop_timer(self, name):
        if self.timers:
            self.timers(name).stop()

    def log_timers(self, name_list):
        if self.timers:
            self.timers.log(name_list)


class SwapTensor(object):
    """Swap Tensor Details"""

    def __init__(self, t, path):
        self.tensor = t
        t.swap_tensor = True
        self.size = t.size()
        self.device = t.device
        self.path = path
        self.is_cpu_tensor = not self.tensor.data.is_cuda
        if self.is_cpu_tensor and not self.tensor.data.is_pinned():
            self.tensor.data = t.data.pin_memory()

    def allocate_memory(self, storage=None):
        if storage is None:
            if self.device.type == 'cpu':
                self.tensor.data = torch.zeros(self.size, device=self.device).pin_memory()
            else:
                self.tensor.data = torch.zeros(self.size, device=self.device)
        else:
            self.tensor.data = storage

    def release_memory(self):
        self.tensor.data = torch.Tensor()

    def __str__(self):
        return f'DeepSpeed.SwapTensor(id={id(self.tensor)}, size={self.size}, device={self.device}, path={self.path}'


class TensorSwapper(object):
    """Mechanism for swapping tensors to/from CPU/GPU memory to storage device"""
    def __init__(self, folder, aio_config, timers=None):
        self.swap_folder = folder
        os.makedirs(self.swap_folder, exist_ok=True)
        self.swap_tensors = []
        self.timers = timers
        self.aio_handle = aio_handle(aio_config[AIO_BLOCK_SIZE],
                                     aio_config[AIO_QUEUE_DEPTH],
                                     aio_config[AIO_SINGLE_SUBMIT],
                                     aio_config[AIO_OVERLAP_EVENTS],
                                     aio_config[AIO_THREAD_COUNT])

    def add_tensors(self, tensor_list):
        for t in tensor_list:
            assert torch.is_tensor(t), f'Non-tensor object {t} of type {type(t)} cannot be added to swapper'
            swap_path = os.path.join(self.swap_folder, f'{id(t)}.tensor.swp')
            swap_tensor = SwapTensor(t, swap_path)
            self.swap_tensors.append(swap_tensor)

    def start_timer(self, name):
        if self.timers:
            self.timers(name).start()

    def stop_timer(self, name):
        if self.timers:
            self.timers(name).stop()

    def log_timers(self, name_list):
        if self.timers:
            self.timers.log(name_list)

    def swap_in_tensors(self):
        ALLOC_TIMER = 'swap_allocate'
        READ_TIMER = 'swap_submit_read'
        WAIT_TIMER = 'swap_wait_read'
        for swap_tensor in self.swap_tensors:
            self.start_timer(ALLOC_TIMER)
            swap_tensor.allocate_memory()
            self.stop_timer(ALLOC_TIMER)

            self.start_timer(READ_TIMER)
            self.aio_handle.async_pread(
                swap_tensor.tensor.data,
                swap_tensor.path)
            self.stop_timer(READ_TIMER)

        self.start_timer(WAIT_TIMER)
        self.aio_handle.wait()
        self.stop_timer(WAIT_TIMER)

        self.log_timers([ALLOC_TIMER, READ_TIMER, WAIT_TIMER])

    def swap_out_tensors(self):
        WRITE_TIMER = 'swap_submit_write'
        WAIT_TIMER = 'swap_wait_write'
        FREE_TIMER = 'swap_free'

        for swap_tensor in self.swap_tensors:
            self.start_timer(WRITE_TIMER)
            self.aio_handle.async_pwrite(
                swap_tensor.tensor.data,
                swap_tensor.path)
            self.stop_timer(WRITE_TIMER)

        self.start_timer(WAIT_TIMER)
        self.aio_handle.wait()
        self.stop_timer(WAIT_TIMER)

        self.start_timer(FREE_TIMER)
        for swap_tensor in self.swap_tensors:
            swap_tensor.release_memory()
        self.stop_timer(FREE_TIMER)

        self.log_timers([WRITE_TIMER, WAIT_TIMER, FREE_TIMER])
