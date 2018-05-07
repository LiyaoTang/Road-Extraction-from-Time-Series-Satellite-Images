#!usr/bin

gpu_cnt=0

# md_dir='/g/data1a/v89/lt8626/Result/sklearn'
# for md_name in 'sk-SGD_weight_m5_0_p1_e15_r1' 'sk-SGD_weight_p0_e15_r0' 'sk-SGD_weight_G0.0001_p0_e15_rNone' 'sk-SGD_G0_0001_p32_e15_r0' 'sk-SGD_G0_001_p16_e15_r1' 'sk-SGD_G0_001_p8_e15_r0' 'sk-SGD_G0_001_p0_e15_r0'; do

# 	echo $md_dir $md_name
# 	python Classifier-Report.py --model_dir ${md_dir} --model_name ${md_name} --gpu ${gpu_cnt} > ./Log/Classifier_Report/${md_name} 2>&1 &
# done

# wait

 # 'Incep_1-32;3-32|1-64;3-64|1-128;3-128_G_weight_p0_e20_r0' 
 # 'Incep_1-32|1-64|1-128_G_weight_bn_p0_e20_r0' 
 # 'Incep_3-32|3-64|3-128_G_weight_p0_e20_r0' 
 # 'Incep_3-32;1-32|3-64;1-64_m_weight_bn_p0_e20_r0'


md_dir='/g/data1a/v89/lt8626/Result/Inception/'
for md_name in 'Incep_1-32;3-32|1-64;3-64|1-128;3-128_G_weight_bn_p0_e20_r0' 'Incep_1-32|1-64|1-128_G_weight_p0_e20_r0' 'Incep_3-32|3-64|3-128_G_weight_bn_p0_e20_r0' 'Incep_3-32;1-32|3-64;1-64_G_weight_bn_p0_e20_r0' 'Incep_3-32;1-32|3-64;1-64_G_weight_p0_e20_r0'; do

	echo $md_dir $md_name
	python Classifier-Report.py --model_dir ${md_dir} --model_name ${md_name} --gpu ${gpu_cnt} > ./Log/Classifier_Report/${md_name} 2>&1 &
	gpu_cnt=$((gpu_cnt+1))
done

wait
