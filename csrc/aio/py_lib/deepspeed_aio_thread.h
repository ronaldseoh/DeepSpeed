
#include <condition_variable>
#include <memory>
#include "deepspeed_py_aio.h"

struct io_op_desc_t {
    const bool _read_op;
    torch::Tensor _buffer;
    int _fd;
    const std::string _filename;
    const long long int _num_bytes;
    torch::Tensor _cpu_buffer;
    torch::Tensor _contiguous_buffer;

    io_op_desc_t(const bool read_op,
                 const torch::Tensor& buffer,
                 const int fd,
                 const char* filename,
                 const long long int num_bytes);

    char* data_ptr() const;
    void fini();
};

struct thread_sync_t {
    std::mutex _mutex;
    std::condition_variable _cond_var;
};

struct deepspeed_aio_thread_t {
    const int _tid;
    deepspeed_aio_config_t& _aio_config;

    std::shared_ptr<struct io_op_desc_t> _next_io_op;
    std::unique_ptr<struct aio_context> _aio_ctxt;

    bool _time_to_exit;
    bool _work_completed;

    struct thread_sync_t _work_sync;
    struct thread_sync_t _complete_sync;

    deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config);

    ~deepspeed_aio_thread_t();

    void run();
};
