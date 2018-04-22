#!/bin/bash

# motor_trunk_pri_sec_tert=0-1-2-3-4
motor_trunk_pri_sec_tert_uncl_track=0-1-2-3-4-5-6
# "motor", "trunk", "pri", "sec", "tert", "uncl", "track", # 0-6
# "res", "serv", "road", "livi", # 7-10

# # center pixel
# for RD in ${motor_trunk_pri_sec_tert} ; do
# 	python Data_Prep-center.py -t ${RD}
# done

# # segment
# for RD in ${motor_trunk_pri_sec_tert_uncl_track} ; do #  ${motor_trunk_pri_sec_tert}
# 	python Data_Prep-seg.py -t ${RD} 
# done


Scene_Dir="../Data/090085/"
Road_Dir="${Scene_Dir}Road_Data/motor_trunk_pri_sec_tert_uncl_track/"
Sliced_Road="${Road_Dir}img_split.h5"
Split_Line="${Road_Dir}img_split_lines.h5"
for SCENE in `cd ${Scene_Dir} && ls 090085_2*.h5`; do
	echo $SCENE
	python Data_Prep-center.py --sliced_road_data ${Sliced_Road}  --image_dir ${Scene_Dir} --image_name ${SCENE} --split_line_path ${Split_Line} --save_dir ${Road_Dir}
	python Data_Prep-seg.py --sliced_road_data ${Sliced_Road}  --image_dir ${Scene_Dir} --image_name ${SCENE} --split_line_path ${Split_Line} --save_dir ${Road_Dir}
done