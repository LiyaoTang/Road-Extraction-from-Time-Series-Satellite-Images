#!/bin/bash

R=../../Data/090085/Road_Data/
H=/patch_set.h5
# motor_trunk_pri motor_trunk_pri_sec_tert_uncl  motor_trunk_pri_sec_tert_uncl_track motor_trunk_pri_sec_tert_uncl_track_res_serv_road_livi tert_uncl_track
# sec_tert_uncl_track
# motor_trunk_pri_sec_res_serv_road_livi motor_trunk_pri_res_serv_road_livi
for path in motor_trunk_pri_sec_res_serv_road_livi motor_trunk_pri_res_serv_road_livi; do
	P=${R}${path}${H}
	ls ${P}
	python NN.py --train ${P} > ./Result/${path} 2>&1
done

