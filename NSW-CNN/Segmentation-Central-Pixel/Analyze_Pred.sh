#!usr/bin

pred_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/
weight=0.3

# for file in `cd ${pred_dir} && ls *.h5`
for file in sk-SGD_weight_p0_e15_r0_pred.h5 sk-SGD_weight_p0_e15_r1_pred.h5 \
			sk-SGD_weight_p8_e15_r0_pred.h5 sk-SGD_weight_p8_e15_r1_pred.h5 \
			sk-SGD_weight_p16_e15_r0_pred.h5 sk-SGD_weight_p16_e15_r1_pred.h5

do
	# echo ${pred_dir}${file} # test
	python Analyze_Pred.py --path ${pred_dir} --name ${file} --pred_weight ${weight} --analyze_CV
	wait
done
