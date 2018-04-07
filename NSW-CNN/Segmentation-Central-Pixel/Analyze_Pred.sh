#!usr/bin

weight=0.3

pred_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/
save_dir=${pred_dir}Pred_Map

for file in `cd ${pred_dir} && ls *0_001*p16*.h5`
do
	echo ${pred_dir}${file} # test

	# path + name => get the h5 file; save => save the pred in png (name inherit the --name opt)
	# python Analyze_Pred.py --path ${pred_dir} --name ${file} --pred_weight ${weight} --save ${save_dir} --analyze_train # --analyze_CV
done