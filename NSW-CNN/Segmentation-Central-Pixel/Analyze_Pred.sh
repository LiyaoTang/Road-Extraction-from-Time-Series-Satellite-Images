#!usr/bin

weight=0.3

pred_dir=./Result/motor_trunk_pri_sec_tert_uncl_track/sklearn/
save_dir=${pred_dir}Pred_Map

# `cd ${pred_dir} && ls *.h5`
for file in sk-SGD_weight_m10_0_p8_e15_r0_pred.h5 sk-SGD_weight_m10_0_p8_e15_r1_pred.h5 ; do

    sub_pred_name=${save_dir}/${file%".h5"}
    if [ ! "$(ls -A ${sub_pred_name} 2>/dev/null)" ]; then
		echo ${pred_dir}${file} # test

		# path + name => get the h5 file; save => save the pred in png (name inherit the --name opt)
		python Analyze_Pred.py --path ${pred_dir} --name ${file} --pred_weight ${weight} --save ${save_dir} --analyze_train --analyze_CV  --print_log --pred_weight 0.5
    else
        printf "%-90s %s\n" "skip ${sub_pred_name}" "- exists"
    fi
done