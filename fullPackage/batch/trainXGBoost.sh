#!/bin/bash
python3 ../trainModel.py -m xgboost -o ge -validation regular -run 115
python3 ../trainModel.py -m xgboost -o ge -validation drug -run 115
python3 ../trainModel.py -m xgboost -o ge -validation cell -run 115


Echo