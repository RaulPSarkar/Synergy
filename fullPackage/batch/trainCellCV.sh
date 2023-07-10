#!/bin/bash
python ../trainModel.py -m dlFull -run 98 -validation regular
python ../trainModel.py -m dlCoeffs -run 97 -validation regular
python ../trainModel.py -m dlFull -run 96 -validation drug
python ../trainModel.py -m dlCoeffs -run 95 -validation drug



Echo