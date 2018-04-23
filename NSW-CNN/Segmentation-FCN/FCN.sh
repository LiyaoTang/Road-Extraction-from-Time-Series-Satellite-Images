#!usr/bin

save_dir=./Result/FCN/
job_cnt=0
gpu_cnt=0

epoch=20
POS=0
for CONV in "16-32-64-128"; do
    for OUT_CONV in "3"; do
        for catIn in "1-16;1-32"; do
            for NORM in m G ; do
                for RAND in 0; do

                    # weighted
                    name=FCN_${CONV}_${OUT_CONV}_cat${catIn}_weight_${NORM}_p${POS}_e${epoch}_r${RAND}
                    echo ${name}
                    python FCN.py --conv ${CONV} --output_conv ${OUT_CONV} --concat_input ${catIn} --norm ${NORM} --pos ${POS} --save ${save_dir} --record_summary --rand ${RAND} --gpu ${gpu_cnt} --epoch ${epoch} >./Log/FCN/${name}  2>&1 &
                    gpu_cnt=$((gpu_cnt+1))

                    # weighted, batch norm
                    name=FCN_${CONV}_${OUT_CONV}_cat${catIn}_weight_bn_${NORM}_p${POS}_e${epoch}_r${RAND}
                    echo ${name}
                    python FCN.py --conv ${CONV} --output_conv ${OUT_CONV} --concat_input ${catIn} --norm ${NORM} --pos ${POS} --save ${save_dir} --record_summary --rand ${RAND} --gpu ${gpu_cnt} --epoch ${epoch} --use_batch_norm >./Log/FCN/${name}  2>&1 &
                    gpu_cnt=$((gpu_cnt+1))

                    sleep 10m
                    job_cnt=$((job_cnt+2))
                    if [ $(($job_cnt%4)) -eq 0 ]; then
                        gpu_cnt=0
                        wait
                    fi
                done
            done
        done
    done
done
