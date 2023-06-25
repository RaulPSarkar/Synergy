#!/bin/bash
python ../trainSKLearn.py -m lgbm -f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/
python ../trainSKLearn.py -m svr -f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/
python ../trainSKLearn.py -m xgboost -f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/
python ../trainSKLearn.py -m base-f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/
python ../trainSKLearn.py -m rf f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/
python ../trainSKLearn.py -m ridge f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/
python ../trainSKLearn.py -m en -trials 20 -f ../datasets/smiles2shuffledfingerprints.csv -run 99 -output ../predictionsShuffled/

Echo