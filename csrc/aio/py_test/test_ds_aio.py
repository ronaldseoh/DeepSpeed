import os
import torch
import argparse
import time
import sys
from multiprocessing import Pool
from deepspeed_aio import deepspeed_aio_read, deepspeed_aio_write, aio_handle

GB_DIVISOR = 1024**3


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--read_file', type=str, default=None, help='Read file.')

    parser.add_argument('--write_file', type=str, default=None, help='Write file.')

    parser.add_argument('--write_size',
                        type=str,
                        default=None,
                        help='Number of bytes to write.')

    parser.add_argument('--block_size', type=str, default='1M', help='I/O block size.')

    parser.add_argument('--queue_depth', type=int, default=32, help='I/O queue depth.')

    parser.add_argument('--threads',
                        type=int,
                        default=1,
                        help='Thread parallelism count.')

    parser.add_argument(
        '--single_submit',
        action='store_true',
        help=
        'Submit I/O requests in singles (default is submit queue_depth amount at once.).'
    )

    parser.add_argument('--overlap_events',
                        action='store_true',
                        help='Overlap I/O submission and completion requests.')

    parser.add_argument('--validate',
                        action='store_true',
                        help='Perform validation in library.')

    parser.add_argument('--handle', action='store_true', help='Use AIO handle.')

    parser.add_argument('--loops',
                        type=int,
                        default=1,
                        help='Count of operation repetitions')

    parser.add_argument('--io_parallel',
                        type=int,
                        default=None,
                        help='Per iop parallelism')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def refine_integer_value(value):
    unit_dict = {'K': 1024, 'M': 1024**2, 'G': 1024**3}

    if value[-1] in list(unit_dict.keys()):
        int_value = int(value[:-1]) * unit_dict[value[-1]]
        return int_value
    return int(value)


def refine_args(args):
    if args.write_size and type(args.write_size) == str:
        args.write_size = refine_integer_value(args.write_size)

    if args.block_size and type(args.block_size) == str:
        args.block_size = refine_integer_value(args.block_size)


def validate_args(args):
    if args.read_file and not os.path.isfile(args.read_file):
        print(f'args validation error: {args.read_file} not found')
        return False

    return True


def do_read(pool_params):
    args, tid = pool_params
    num_bytes = os.path.getsize(args.read_file)
    print(f'tid {tid}: Allocating read tensor of size {num_bytes} bytes')
    dst_buffer = torch.empty(num_bytes, dtype=torch.uint8, device='cpu').pin_memory()
    print(
        f'tid {tid}: Reading file {args.read_file} of size {num_bytes} bytes into buffer on device {dst_buffer.device}'
    )
    elapsed_sec = 0
    for _ in range(args.loops):
        start_time = time.time()
        deepspeed_aio_read(dst_buffer,
                           args.read_file,
                           args.block_size,
                           args.queue_depth,
                           args.single_submit,
                           args.overlap_events,
                           args.validate)
        end_time = time.time()
        elapsed_sec += end_time - start_time
    print(f'tid {tid}:  read_time(usec) = {elapsed_sec*1e6}')
    return elapsed_sec, num_bytes


def do_write(pool_params):
    args, tid = pool_params
    write_file = f'{args.write_file}.{tid}'
    fp32_bytes = torch.tensor([]).element_size()
    fp32_size = args.write_size // fp32_bytes
    print(f'tid {tid}: Allocating write tensor of size {args.write_size} bytes')
    write_tensor = torch.randn(fp32_size, dtype=torch.float32, device='cpu').pin_memory()
    tensor_bytes = write_tensor.element_size() * write_tensor.numel()
    print(
        f'tid {tid}:  Writing file {write_file} of size {tensor_bytes} bytes from buffer on device {write_tensor.device}'
    )
    elapsed_sec = 0
    for _ in range(args.loops):
        start_time = time.time()
        deepspeed_aio_write(write_tensor,
                            write_file,
                            args.block_size,
                            args.queue_depth,
                            args.single_submit,
                            args.overlap_events,
                            args.validate)
        end_time = time.time()
        elapsed_sec += end_time - start_time
    print(f'tid {tid}: write_time(usec) = {elapsed_sec*1e6}')
    return elapsed_sec, tensor_bytes


def do_handle_read(pool_params):
    args, tid = pool_params
    num_bytes = os.path.getsize(args.read_file)
    print(f'tid {tid}: Allocating read tensor of size {num_bytes} bytes')
    dst_buffer = torch.empty(num_bytes, dtype=torch.uint8, device='cpu').pin_memory()
    print(
        f'tid {tid}: Reading file {args.read_file} of size {num_bytes} bytes into buffer on device {dst_buffer.device}'
    )
    ds_aio_handle = aio_handle(args.block_size,
                               args.queue_depth,
                               args.single_submit,
                               args.overlap_events,
                               1)
    print(f'tid {tid}: created deepspeed aio handle')
    elapsed_sec = 0
    for _ in range(args.loops):
        start_time = time.time()
        ret = ds_aio_handle.read(dst_buffer, args.read_file, args.validate)
        assert ret != -1
        end_time = time.time()
        elapsed_sec += end_time - start_time

    ds_aio_handle.fini()
    print(f'tid {tid}:  read_time(usec) = {elapsed_sec*1e6}')
    return elapsed_sec, num_bytes


def do_parallel_read(pool_params):
    args, tid = pool_params
    num_bytes = os.path.getsize(args.read_file)
    print(f'tid {tid}: Allocating read tensor of size {num_bytes} bytes')
    dst_buffer = torch.empty(num_bytes, dtype=torch.uint8, device='cpu').pin_memory()
    print(
        f'tid {tid}: Reading file {args.read_file} of size {num_bytes} bytes into buffer on device {dst_buffer.device}'
    )
    ds_aio_handle = aio_handle(args.block_size,
                               args.queue_depth,
                               args.single_submit,
                               args.overlap_events,
                               args.io_parallel)
    print(f'tid {tid}: created deepspeed aio handle')
    elapsed_sec = 0
    for _ in range(args.loops):
        start_time = time.time()
        ret = ds_aio_handle.pread(dst_buffer, args.read_file, args.validate, True)
        assert ret != -1
        ds_aio_handle.wait(args.validate)
        end_time = time.time()
        elapsed_sec += end_time - start_time

    ds_aio_handle.fini()
    print(f'tid {tid}:  read_time(usec) = {elapsed_sec*1e6}')
    return elapsed_sec, num_bytes


def do_handle_write(pool_params):
    args, tid = pool_params
    write_file = f'{args.write_file}.{tid}'
    print(f'tid {tid}: Allocating write tensor of size {args.write_size} bytes')
    write_tensor = torch.empty(args.write_size,
                               dtype=torch.uint8,
                               device='cpu').pin_memory()
    print(
        f'tid {tid}:  Writing file {write_file} of size {args.write_size} bytes from buffer on device {write_tensor.device}'
    )

    ds_aio_handle = aio_handle(args.block_size,
                               args.queue_depth,
                               args.single_submit,
                               args.overlap_events,
                               1)
    print(f'tid {tid}: created deepspeed aio handle')

    elapsed_sec = 0
    for _ in range(args.loops):
        start_time = time.time()
        ret = ds_aio_handle.write(write_tensor, write_file, args.validate)
        assert ret != -1
        end_time = time.time()
        elapsed_sec += end_time - start_time

    ds_aio_handle.fini()
    print(f'tid {tid}: write_time(usec) = {elapsed_sec*1e6}')
    return elapsed_sec, args.write_size


def do_parallel_write(pool_params):
    args, tid = pool_params
    write_file = f'{args.write_file}.{tid}'
    print(f'tid {tid}: Allocating write tensor of size {args.write_size} bytes')
    write_tensor = torch.empty(args.write_size,
                               dtype=torch.uint8,
                               device='cpu').pin_memory()
    print(
        f'tid {tid}:  Writing file {write_file} of size {args.write_size} bytes from buffer on device {write_tensor.device}'
    )

    ds_aio_handle = aio_handle(args.block_size,
                               args.queue_depth,
                               args.single_submit,
                               args.overlap_events,
                               args.io_parallel)
    print(f'tid {tid}: created deepspeed aio handle')

    elapsed_sec = 0
    for _ in range(args.loops):
        start_time = time.time()
        ret = ds_aio_handle.pwrite(write_tensor, write_file, args.validate, True)
        assert ret != -1
        ds_aio_handle.wait(args.validate)
        end_time = time.time()
        elapsed_sec += end_time - start_time

    ds_aio_handle.fini()
    print(f'tid {tid}: write_time(usec) = {elapsed_sec*1e6}')
    return elapsed_sec, args.write_size


def do_parallel_io(args, io_function, io_string):
    POOL_SIZE = args.threads
    pool_params = [(args, p) for p in range(POOL_SIZE)]

    with Pool(POOL_SIZE) as p:
        io_perf_results = p.map(io_function, pool_params)

    if None in io_perf_results:
        print(f"Failure in one of {POOL_SIZE} {io_string} processes")
        return

    max_latency_sec = max([sec for sec, _ in io_perf_results])
    total_bytes = sum([num_bytes for _, num_bytes in io_perf_results])
    io_speed_GB = args.loops * total_bytes / max_latency_sec / GB_DIVISOR
    print(f'Total {io_string} Speed = {io_speed_GB} GB/sec')


def main():
    print(f'Testing deepspeed_aio python frontend')

    args = parse_arguments()
    refine_args(args)
    if not validate_args(args):
        quit()

    if args.read_file:
        if args.handle:
            if args.io_parallel:
                read_function = do_parallel_read
            else:
                read_function = do_handle_read
        else:
            read_function = do_read
        do_parallel_io(args, read_function, "Read")

    if args.write_file:
        if args.handle:
            if args.io_parallel:
                write_function = do_parallel_write
            else:
                write_function = do_handle_write
        else:
            write_function = do_write
        do_parallel_io(args, write_function, 'Write')


if __name__ == "__main__":
    main()
