#!/bin/bash
if [[ -z $1 ]]; then
	echo "Usage: $0 <file to write>"
	exit 1
fi
FILE=$1
RUN_SCRIPT=./test_ds_aio.py
SIZE=400M
WRITE_OPT="--write_file ${FILE} --write_size ${SIZE}"
LOG_DIR=/tmp/py_ds_aio_write_${SIZE}B
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
                    LOG="${LOG_DIR}/write_${SIZE}B_${sub}_${ov}_p${p}_d${d}_bs${bs}.txt"
                    cmd="python ${RUN_SCRIPT} ${WRITE_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"
                    echo ${cmd}
                    eval ${cmd}
                    sleep 2
                done
            done
        done
    done
done
