import pytest
import os
import filecmp
import torch

try:
    from deepspeed.ops.aio import aio_handle
except ImportError:
    print('DeepSpeed AIO library is not installed, will skip unit tetts')
    aio_handle = None

MEGA_BYTE = 1024**2
BLOCK_SIZE = MEGA_BYTE
QUEUE_DEPTH = 2
IO_SIZE = 16 * MEGA_BYTE
IO_PARALLEL = 2
AIO_VALIDATE = False


def _skip_if_no_aio():
    try:
        from deepspeed.ops.aio import aio_handle
    except ImportError:
        pytest.skip('Skip these tests until libaio-dev is installed in our docker image')


def _do_ref_write(tmpdir):
    ref_file = os.path.join(tmpdir, '_py_random.pt')
    ref_buffer = os.urandom(IO_SIZE)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    return ref_file, ref_buffer


def _get_test_file_and_buffer(tmpdir, ref_buffer, cuda_device):
    test_file = os.path.join(tmpdir, '_aio_write_random.pt')
    if cuda_device:
        test_buffer = torch.cuda.ByteTensor(list(ref_buffer))
    else:
        test_buffer = torch.ByteTensor(list(ref_buffer)).pin_memory()

    return test_file, test_buffer


def _validate_handle_state(handle, single_submit, overlap_events):
    assert handle.get_single_submit() == single_submit
    assert handle.get_overlap_events() == overlap_events
    assert handle.get_thread_count() == IO_PARALLEL
    assert handle.get_block_size() == BLOCK_SIZE
    assert handle.get_queue_depth() == QUEUE_DEPTH


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
    _skip_if_no_aio()

    ref_file, _ = _do_ref_write(tmpdir)

    aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu').pin_memory()
    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    _validate_handle_state(h, single_submit, overlap_events)

    read_status = h.sync_pread(aio_buffer, ref_file)
    assert read_status == 0

    with open(ref_file, 'rb') as f:
        ref_buffer = list(f.read())
    assert ref_buffer == aio_buffer.tolist()


@pytest.mark.parametrize('single_submit, overlap_events, cuda_device',
                         [(False,
                           False,
                           False),
                          (False,
                           True,
                           False),
                          (True,
                           False,
                           False),
                          (True,
                           True,
                           False),
                          (False,
                           False,
                           True),
                          (True,
                           True,
                           True)])
def test_async_read(tmpdir, single_submit, overlap_events, cuda_device):
    _skip_if_no_aio()

    ref_file, _ = _do_ref_write(tmpdir)

    if cuda_device:
        aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cuda')
    else:
        aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu').pin_memory()

    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    _validate_handle_state(h, single_submit, overlap_events)

    read_status = h.async_pread(aio_buffer, ref_file)
    assert read_status == 0

    wait_status = h.wait(AIO_VALIDATE)
    assert wait_status == 0

    with open(ref_file, 'rb') as f:
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
    _skip_if_no_aio()

    ref_file, ref_buffer = _do_ref_write(tmpdir)

    aio_file, aio_buffer = _get_test_file_and_buffer(tmpdir, ref_buffer, False)

    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    _validate_handle_state(h, single_submit, overlap_events)

    write_status = h.sync_pwrite(aio_buffer, aio_file)
    assert write_status == 0

    assert os.path.isfile(aio_file)

    filecmp.clear_cache()
    assert filecmp.cmp(ref_file, aio_file, shallow=False)


@pytest.mark.parametrize('single_submit, overlap_events, cuda_device',
                         [(False,
                           False,
                           False),
                          (False,
                           True,
                           False),
                          (True,
                           False,
                           False),
                          (True,
                           True,
                           False),
                          (False,
                           False,
                           True),
                          (True,
                           True,
                           True)])
def test_async_write(tmpdir, single_submit, overlap_events, cuda_device):
    _skip_if_no_aio()

    ref_file, ref_buffer = _do_ref_write(tmpdir)

    aio_file, aio_buffer = _get_test_file_and_buffer(tmpdir, ref_buffer, cuda_device)

    h = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
    _validate_handle_state(h, single_submit, overlap_events)

    write_status = h.async_pwrite(aio_buffer, aio_file)
    assert write_status == 0

    wait_status = h.wait(AIO_VALIDATE)
    assert wait_status == 0

    assert os.path.isfile(aio_file)

    filecmp.clear_cache()
    assert filecmp.cmp(ref_file, aio_file, shallow=False)
