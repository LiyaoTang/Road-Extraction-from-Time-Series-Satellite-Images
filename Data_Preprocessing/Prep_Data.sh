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

for SCENE in ``; do
	python Data_Prep-seg.py --sliced_road_data
done