#!/bin/bash
python3 ../trainSKLearn.py -m lgbm -o proteomics -run 100
python3 ../trainSKLearn.py -m lgbm -o ge -run 100
python3 ../trainSKLearn.py -m dlCoeffs -run 101
python3 ../trainSKLearn.py -m xgboost -trials 12 -run 100


Echo