#!/bin/bash
if [[ -z $1 ]]; then
	echo "Usage: $0 <file to read>"
	exit 1
fi

FILE=$1
RUN_SCRIPT=./test_ds_aio.py
READ_OPT="--read_file ${FILE}"
LOG_DIR=/tmp/py_ds_aio_read
mkdir -p ${LOG_DIR}
rm -f ${LOG_DIR}/*
for sub in single block; do
    if [[ $sub == "single" ]]; then
        sub_opt="--single_submit"
    else
        sub_opt=""
    fi
    for ov in overlap sequential; do
        if [[ $ov == "overlap" ]]; then
            ov_opt="--overlap_events"
        else
            ov_opt=""
        fi
        for p in 1 2 4 8 16 32; do
            for d in 1 2 4 8 16 32; do
                for bs in 128K 256K 512K 1M; do
                    SCHED_OPTS="${sub_opt} ${ov_opt} --handle --threads 1"
                    OPTS="--io_parallel ${p} --queue_depth ${d} --block_size ${bs}"
                    LOG="${LOG_DIR}/read_${sub}_${ov}_p${p}_d${d}_bs${bs}.txt"
                    cmd="python ${RUN_SCRIPT} ${READ_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"
                    echo ${cmd}
                    eval ${cmd}
                    sleep 2
                done
            done
        done
    done
done
