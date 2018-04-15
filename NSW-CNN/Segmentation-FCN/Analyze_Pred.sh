#!usr/bin

weight=0.3

pred_dir=./Result/Inception/
save_dir=${pred_dir}Pred_Map

size=128
step=18

for sub_dir in `cd ${pred_dir} && ls`; do

    for file in `cd ${pred_dir}${sub_dir} && ls Incep_3-8\;5-8\;1-8\|3-4\;1-4_*.h5 2>/dev/null`; do
        path=${pred_dir}${sub_dir}

        sub_pred_name=${save_dir}/${file%".h5"}
        if [ ! "$(ls -A ${sub_pred_name}*CV* 2>/dev/null)" ]; then
            echo processing ${sub_pred_name}
            # path + name => get the h5 file; save => save the pred in png (name inherit the --name opt)
            python Analyze_Pred.py --path ${path} --name ${file} --pred_weight ${weight} --save ${save_dir} --step ${step} --size ${size} \
            --analyze_CV # CV train
        else
            printf "%-90s %s\n" "skip ${sub_pred_name}" "- exists"
        fi
    done
done