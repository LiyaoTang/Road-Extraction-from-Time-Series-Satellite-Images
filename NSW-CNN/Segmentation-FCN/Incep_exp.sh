#!usr/bin

save_dir=./Result/Inception/
epoch=20

for CONV in "0" "3-16;5-8;1-32" "3-16;5-8;1-32|3-8;1-16"; do # not tested: "3-8;5-8;1-8|3-4;1-4"
	for POS in 0 8; do
		for RAND in 0 1; do
			for NORM in m G; do
				name=Incep_${CONV}_${NORM}_weight_p0_e${epoch}_r${RAND}
				python FCN-inception.py --conv ${CONV} --norm ${NORM} --save ${save_dir} --rand ${RAND} --gpu ${RAND} --epoch ${epoch} >./Log/Inception/${name}  2>&1

				name=Incep_${CONV}_${NORM}_weight_bn_p0_e${epoch}_r${RAND}
				python FCN-inception.py --conv ${CONV} --norm ${NORM} --save ${save_dir} --rand ${RAND} --gpu ${RAND} --epoch ${epoch} --use_batch_norm >./Log/Inception/${name}  2>&1
			done
		done
	done
done