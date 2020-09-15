#include <torch/extension.h>
#include "deepspeed_py_aio_handle.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("aio_read", &deepspeed_py_aio_read, "DeepSpeed Asynchornous I/O Read");

    m.def("aio_write", &deepspeed_py_aio_write, "DeepSpeed Asynchornous I/O Write");

    py::class_<deepspeed_aio_handle_t>(m, "aio_handle")
        .def(py::init<const int, const int, const bool, const bool, const int>())
        .def("get_block_size", &deepspeed_aio_handle_t::get_block_size)
        .def("get_queue_depth", &deepspeed_aio_handle_t::get_queue_depth)
        .def("get_single_submit", &deepspeed_aio_handle_t::get_single_submit)
        .def("get_overlap_events", &deepspeed_aio_handle_t::get_overlap_events)
        .def("read", &deepspeed_aio_handle_t::read)
        .def("write", &deepspeed_aio_handle_t::write)
        .def("pread", &deepspeed_aio_handle_t::pread)
        .def("pwrite", &deepspeed_aio_handle_t::pwrite)
        .def("wait", &deepspeed_aio_handle_t::wait);
}
