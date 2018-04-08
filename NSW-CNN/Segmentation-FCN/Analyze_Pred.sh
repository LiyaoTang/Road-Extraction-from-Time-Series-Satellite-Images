#!usr/bin

weight=0.3

pred_dir=./Result/Inception/
save_dir=${pred_dir}Pred_Map

size=128
step=18

for sub_dir in `cd ${pred_dir} && ls`; do
    for file in `cd ${pred_dir}${sub_dir} && ls *.h5 2>/dev/null`; do
        path=${pred_dir}${sub_dir}

        sub_pred_name=${save_dir}/${file%".h5"}
        if [ ! "$(ls -A ${sub_pred_name}* 2>/dev/null)" ]; then
            # path + name => get the h5 file; save => save the pred in png (name inherit the --name opt)
            python Analyze_Pred.py --path ${path} --name ${file} --pred_weight ${weight} --save ${save_dir} --step ${step} --size ${size} \
            --analyze_train # --analyze_CV
        else
            printf "%-90s %s\n" "skip ${sub_pred_name}" "- exists"
        fi
    done
done