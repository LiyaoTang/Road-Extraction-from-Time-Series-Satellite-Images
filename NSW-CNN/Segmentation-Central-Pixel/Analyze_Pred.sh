#!usr/bin

pred_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/
weight=0.3

for file in `cd ${pred_dir} && ls *n*.h5`
do
	# echo ${pred_dir}${file} # test
	python Analyze_Pred.py --path ${pred_dir} --name ${file} --pred_weight ${weight} --analyze_CV
	wait
done
