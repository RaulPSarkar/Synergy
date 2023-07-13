#!/bin/bash

python3 ../trainModel.py -m rf -run 11 -validation drug
python3 ../trainModel.py -m rf-run 11 -validation cell
python3 ../trainModel.py -m xgboost -run 99 -validation drug
python3 ../trainModel.py -m xgboost -run 99 -validation cell
python3 ../trainModel.py -m lgbm -run 99 -validation drug
python3 ../trainModel.py -m lgbm -run 99 -validation cell
python3 ../trainModel.py -m en -run 3 -validation drug
python3 ../trainModel.py -m en -run 3 -validation cell
python3 ../trainModel.py -m svr -run 1 -validation drug
python3 ../trainModel.py -m svr -run 1 -validation cell
python3 ../trainModel.py -m dl -run 1 -validation drug
python3 ../trainModel.py -m dl -run 1 -validation cell

Echo