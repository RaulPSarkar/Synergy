#!/bin/bash
python ../trainSKLearn.py -m lgbm -run 100
python ../trainSKLearn.py -m dlCoeffs -run 100
python ../trainSKLearn.py -m rf -run 100
python ../trainSKLearn.py -m xgboost -trials 12 -run 100


Echo