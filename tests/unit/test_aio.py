import pytest
import os
import filecmp
import torch
from deepspeed_aio import aio_handle

MEGA_BYTE = 1024**2
BLOCK_SIZE = MEGA_BYTE
QUEUE_DEPTH = 2
IO_SIZE = 16 * MEGA_BYTE
IO_PARALLEL = 2
AIO_VALIDATE = False


@pytest.mark.parametrize('single_submit, overlap_events',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_parallel_read(tmpdir, single_submit, overlap_events):
    test_file = os.path.join(tmpdir, '_aio_random.pt')
    with open(test_file, 'wb') as f:
        f.write(os.urandom(IO_SIZE))

    aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu').pin_memory()
    asynchronous_read = False
    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    read_status = h.pread(aio_buffer, test_file, AIO_VALIDATE, asynchronous_read)
    assert read_status == 0

    with open(test_file, 'rb') as f:
        ref_buffer = list(f.read())

    assert ref_buffer == aio_buffer.tolist()


@pytest.mark.parametrize('single_submit, overlap_events',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_async_read(tmpdir, single_submit, overlap_events):
    test_file = os.path.join(tmpdir, '_aio_random.pt')
    with open(test_file, 'wb') as f:
        f.write(os.urandom(IO_SIZE))

    aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu').pin_memory()
    asynchronous_read = True
    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    read_status = h.pread(aio_buffer, test_file, AIO_VALIDATE, asynchronous_read)
    assert read_status == 0

    wait_status = h.wait(AIO_VALIDATE)
    assert wait_status == 0

    with open(test_file, 'rb') as f:
        ref_buffer = list(f.read())

    assert ref_buffer == aio_buffer.tolist()


@pytest.mark.parametrize('single_submit, overlap_events',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_parallel_write(tmpdir, single_submit, overlap_events):

    ref_file = os.path.join(tmpdir, '_py_write_random.pt')
    ref_buffer = os.urandom(IO_SIZE)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    test_file = os.path.join(tmpdir, '_aio_write_random.pt')
    aio_buffer = torch.ByteTensor(list(ref_buffer)).pin_memory()
    asynchronous_write = False

    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    write_status = h.pwrite(aio_buffer, test_file, AIO_VALIDATE, asynchronous_write)
    assert write_status == 0

    assert os.path.isfile(test_file)

    filecmp.clear_cache()
    assert filecmp.cmp(ref_file, test_file, shallow=False)


@pytest.mark.parametrize('single_submit, overlap_events',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_async_write(tmpdir, single_submit, overlap_events):

    ref_file = os.path.join(tmpdir, '_py_write_random.pt')
    ref_buffer = os.urandom(IO_SIZE)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    test_file = os.path.join(tmpdir, '_aio_write_random.pt')
    aio_buffer = torch.ByteTensor(list(ref_buffer)).pin_memory()
    asynchronous_write = True

    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    write_status = h.pwrite(aio_buffer, test_file, AIO_VALIDATE, asynchronous_write)
    assert write_status == 0

    wait_status = h.wait(AIO_VALIDATE)
    assert wait_status == 0

    assert os.path.isfile(test_file)

    filecmp.clear_cache()
    assert filecmp.cmp(ref_file, test_file, shallow=False)
