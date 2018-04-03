#!/bin/bash

save_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/

for RAND in 0 1; do
    for NORM in m G; do
        for POS in 0 8; do
            norm_param=0.001

            name=SGD_weight_${NORM}${norm_param}_p${POS}_e15_r${RAND}
            python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm_param ${norm_param} --save $save_dir > ./Log/sklearn/${name} 2>&1 &

            name=SGD_${NORM}${norm_param}_p${POS}_e15_r${RAND}
            python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm_param ${norm_param} --save $save_dir --not_weight > ./Log/sklearn/${name} 2>&1 &
            
            sleep 10m
        done
    done

    wait
done

# for i in `seq 0 1 9`
# do

#         name=SGD_weight_p0_n0_0001-0_01_sample_r${i}
# 	python Logistic-Reg.py --rand $i --pos 0 --norm_param 0.0001-0.010 --sample_norm 4 --save $save_dir > ./Log/sklearn/${name} 2>&1 &

#         wait
# done

# wait

# for i in `seq 0 1 9`
# do
#         name=SGD_weight_p16_n5-20_sample_r${i}
#         python Logistic-Reg.py --rand $i --pos 16 --norm_param 5-20 --sample_norm 1 --save $save_dir > ./Log/sklearn/${name} 2>&1 &
#         if [ $i == 4 ]
#         then
#                 wait
#         fi
# done

# wait

# for i in `seq 0 1 9`
# do
#         name=SGD_weight_p0_nG5-20_sample_r${i}
#         python Logistic-Reg.py --rand $i --pos 0 --norm G --norm_param 5-20 --sample_norm 1 --save $save_dir > ./Log/sklearn/${name} 2>&1 &
#         if [ $i == 4 ]
#         then
#                 wait
#         fi
# done

