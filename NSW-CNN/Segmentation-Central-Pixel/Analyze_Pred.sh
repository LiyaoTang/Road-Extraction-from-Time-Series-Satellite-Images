#!usr/bin

weight=0.3

#pred_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/mean/
#save_dir=${pred_dir}Pred_Map
pred_dir=../../..
save_dir=../../..
for file in sk-SGD_weight_p8_n100_01_e15_r1_pred.h5 # `cd ${pred_dir} && ls *.h5`
do
	# echo ${pred_dir}${file} # test
	python Analyze_Pred.py --path ${pred_dir} --name ${file} --pred_weight ${weight} --save ${save_dir} --analyze_CV
done