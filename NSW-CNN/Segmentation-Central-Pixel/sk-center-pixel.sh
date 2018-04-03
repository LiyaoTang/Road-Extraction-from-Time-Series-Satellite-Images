#!/bin/bash

py_dir=./Sat-Img-in-Time/NSW-CNN/Segmentation-Central-Pixel/
cd $py_dir

save_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/


for i in `seq 0 1 10`
do

	python Logistic-Reg.py --rand $i --pos 8 --norm_param 0.0001-0.010 --sample_norm 4 --save $save_dir > ./Log/sklearn/SGD_weight_p8_n0_0001-0_01_sample_r${i} 2>&1 &
	if [ $i == 3 ] || [ $i == 7 ]
	then
		wait
	fi
done

wait

for i in `seq 0 1 10`
do
        python Logistic-Reg.py --rand $i --pos 16 --norm_param 5-20 --sample_norm 1 --save $save_dir > ./Log/sklearn/SGD_weight_p16_n5-20_sample_r${i} 2>&1 &
        if [ $i == 3 ] || [ $i == 7 ]
        then
                wait
        fi
done

wait

for i in `seq 0 1 10`
do
        python Logistic-Reg.py --rand $i --pos 0 --norm G --norm_param 5-20 --sample_norm 1 --save $save_dir > ./Log/sklearn/SGD_weight_p0_nG5-20_sample_r${i} 2>&1 &
        if [ $i == 3 ] || [ $i == 7 ]
        then
                wait
        fi
done

