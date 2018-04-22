#!/bin/bash

save_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/
job_cnt=0

# for RAND in 0 1; do
#     for NORM in m G; do
#         for NORM_PARM in 0.001; do
#             for POS in 16; do # tested: 0 8
                
#                 job_cnt=$((job_cnt+2))

#                 name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#                 python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &
                
#                 name=SGD_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#                 python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir --not_weight > ./Log/sklearn/${name} 2>&1 &
                
#                 sleep 10m

#                 if [ $job_cnt -eq 6 ] || [ $job_cnt -eq 12 ]; then
#                     wait
#                 fi
#             done
#         done
#     done
# done

# result:
# need smaller pos (0-8) , larger norm_param

# for RAND in 0 1; do
#     for NORM in m G; do
#         for NORM_PARM in 0.01 0.1 1 10; do
#             for POS in 8; do # tested: 0 8
                
#                 job_cnt=$((job_cnt+1))

#                 name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#                 python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &                
#                 sleep 10m

#                 echo $name

#                 if [ $(($job_cnt%6)) -eq 0 ]; then
#                     wait
#                 fi
#             done
#         done
#     done
# done

# result:
# largfer norm_param does help with restricting it from predicting all 1 (roads) 
#   => norm = G: reduced in a global sense => reduce the pred_prob overall
#   => norm = m: better result => does filtering out montain area

# for RAND in 0 1; do
#     for NORM in m G; do
#         for NORM_PARM in 0.001; do
#             for POS in 2 4 6; do # tested: 0 8
                
#                 job_cnt=$((job_cnt+1))

#                 name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#                 python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &                
#                 sleep 10m

#                 echo $name

#                 if [ $(($job_cnt%6)) -eq 0 ]; then
#                     wait
#                 fi
#             done
#         done
#     done
# done

# result:
# 2,4,6 not good under a small norm_param => even pos=2 causes most pred to be 1 (for both norm=G, m)                

# for RAND in 0 1; do
#     for NORM in m; do
#         for NORM_PARM in 0.01 0.1 0.5 1 5 10; do
#             for POS in 0 1; do
                
#                 job_cnt=$((job_cnt+1))

#                 name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#                 python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &                
#                 sleep 10m

#                 echo $name

#                 if [ $(($job_cnt%6)) -eq 0 ]; then
#                     wait
#                 fi
#             done
#         done
#     done
# done

# result: little / no upsampling & there norm_param search
# the coefficient under large regularization is much smaller, yet gives similar prediction => too small coefficents give blur result (image not sharp at all)


# RAND=0
# POS=0
# NORM=m
# for NORM_PARM in 0.02 0.05 0.08; do
#     name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#     python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &                
#     sleep 10m
# done

# result: smaller regularization the better => to have more difference between positive pred and negative pred

# RAND=0
# POS=0
# NORM=G
# for NORM_PARM in 0.1 0.5 1; do
#     name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
#     python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &                
#     sleep 10m
# done

# result: smaller regularization the better => small input with small coefficient cannot make difference in decision

RAND=None
POS=0
NORM_PARM=0.0001
for NORM in m G; do
    name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
    echo $name
    python Logistic-Reg.py --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &                

    wait

    name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r1
    echo $name
    python Logistic-Reg.py --rand 1 --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir > ./Log/sklearn/${name} 2>&1 &
    
    wait
    #sleep 10m
done

# test for no normalization
POS=0
for RAND in 0 1; do
    name=SGD_weight_p${POS}_e15_r${RAND}
    echo $name
    python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir --not_norm > ./Log/sklearn/${name} 2>&1 &
    wait
done

wait

# test for 0.5 upsampling (pos=32)
RAND=0
for NORM_PARM in 0.0001 0.001; do
    for NORM in m G; do
        for POS in 32; do
            name=SGD_weight_${NORM}${NORM_PARM}_p${POS}_e15_r${RAND}
            echo $name
            python Logistic-Reg.py --rand ${RAND} --pos ${POS} --norm ${NORM} --norm_param ${NORM_PARM} --save $save_dir --not_weight > ./Log/sklearn/${name} 2>&1 &
            wait
        done
    done
done
