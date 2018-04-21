#!usr/bin

save_dir=./Result/Inception/
epoch=30

# for CONV in "3-16;5-8;1-32|3-8;1-16" "3-8;5-8;1-8|3-4;1-4"; do
#     for RAND in 0; do
#         for NORM in m G; do
#             name=Incep_${CONV}_${NORM}_weight_p${POS}_e${epoch}_r${RAND}
#             python FCN-inception.py --conv ${CONV} --norm ${NORM} --pos ${POS} --save ${save_dir} --record_summary --rand ${RAND} --gpu ${RAND} --epoch ${epoch} >./Log/Inception/${name}  2>&1

#             name=Incep_${CONV}_${NORM}_weight_bn_p${POS}_e${epoch}_r${RAND}
#             python FCN-inception.py --conv ${CONV} --norm ${NORM} --pos ${POS} --save ${save_dir} --record_summary --rand ${RAND} --gpu ${RAND} --epoch ${epoch} --use_batch_norm >./Log/Inception/${name}  2>&1
#         done
#     done
# done

# result on samll dataset: 

# without upsampling pos, as using batch_size=1:
#   from ROC, using only pixel-level spectral info (CONV=0) with different setting = random gussing (though cross entropy is minimized in training)
#   models with 1 layer generally worse than models with 2 layers
#   under same setting, "3-16;5-8;1-32|3-8;1-16", "3-8;5-8;1-8|3-4;1-4" did learn something

POS=0
for CONV in "3-32|3-64" "3-32;1-32|3-64;1-64"; do
    for RAND in 0; do
        for NORM in m G; do
            name=Incep_${CONV}_${NORM}_weight_p${POS}_e${epoch}_r${RAND}
            python FCN-inception.py --conv ${CONV} --norm ${NORM} --pos ${POS} --save ${save_dir} --record_summary --rand ${RAND} --gpu ${RAND} --epoch ${epoch} >./Log/Inception/${name}  2>&1
            name=Incep_${CONV}_${NORM}_weight_noB_p${POS}_e${epoch}_r${RAND}
            python FCN-inception.py --conv ${CONV} --norm ${NORM} --pos ${POS} --save ${save_dir} --record_summary --rand ${RAND} --gpu ${RAND} --epoch ${epoch} --use_batch_norm >./Log/Inception/${name}  2>&1
        done
    done
done
