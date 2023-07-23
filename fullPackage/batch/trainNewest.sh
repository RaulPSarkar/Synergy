#!/bin/bash
python ../trainSKLearn.py -m lgbm -run 99
python ../trainSKLearn.py -m rf -run 99
python ../trainSKLearn.py -m xgboost -run 99


python ../trainSKLearn.py -m base -run 99
python ../trainSKLearn.py -m ridge -run 99
python ../trainSKLearn.py -m en -trials 20 -run 99

Echo