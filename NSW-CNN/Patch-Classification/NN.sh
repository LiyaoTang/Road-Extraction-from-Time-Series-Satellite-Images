#!/bin/bash

# 0 motorway.tif
# 1 trunk.tif
# 2 primary.tif
# 3 secondary.tif
# 4 tertiary.tif
# 5 unclassified.tif
# 6 track.tif
# 7 residential.tif
# 8 service.tif
# 9 road.tif
# 10 living_street.tif

# 1. motorway, trunk, primary
# 2. motorway, trunk, primary, secondary 0.92 0.59 \\
# 3. motorway, trunk, primary, secondary, tertiary 0.88 0.68 \\
# 4. motorway, trunk, primary, secondary, tertiary, unclassified 0.77  0.11 \\
# 5. motorway, trunk, primary, secondary, tertiary, unclassified, track 0.66 0.28 \\
# 6. motorway, trunk, primary, secondary, tertiary, unclassified, track, residential, service, road, living street 0.65 0.27 \\
# 7. secondary, tertiary, unclassified, track 0.67  0.19 \\
# 8. tertiary, unclassified, track 0.69  0.14 \\
# 9. residential, service, road, living street 0.96 0.84 \\
# 10. motorway, trunk, primary, residential, service, road, living street 0.94 0.15 \\
# 11. motorway, trunk, primary, secondary, residential, service, road, living street 0.91 0.13 \\

for RT in "0-1-2" "0-1-2-3" "0-1-2-3-4" "0-1-2-3-4-5" "0-1-2-3-4-5-6" "0-1-2-3-4-5-6-7-8-9-10" "3-4-5-6" "4-5-6" "7-8-9-10" "0-1-2-7-8-9-10" "0-1-2-3-7-8-9-10"; do
	echo $RT
	python NN.py --road_type ${RT} > ./Result/${RT}_rst 2>&1
done

