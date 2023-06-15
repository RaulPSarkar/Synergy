# Synergy

Running processGDSC.py will concatenate pancreas_anchor_combo.csv, colon_anchor_combo.csv, and breast_anchor_combo.csv, and generates a dataset called "dataset.csv" with an added SMILES A and SMILES B columns. It keeps only rows with known SMILES for both drugs (only 1 drug doesn't have a known SMILES).
A cached version of drug names to SMILES is present at drug2smiles.txt and is used for this process.
Afterwards, it selects between duplicates (i.e. same drug pair-cell line triplets with different concentrations) using the Mahalanobis distance. The output from this script is processedCRISPR.csv, and it contains all the experiments for which the cell line is known. It includes the X columns [Drug Smiles A, Drug Smiles B, Cell Line], and the Y Columns [Delta IC50, Delta Emax]
The rows are shuffled before saving to the .csv, so that they don't have to be shuffled during training. This is to ensure that all model predictions have the experiments in the same order (for ensemble).

pip install pubchempy
pip install rdkit
pip install cirpy
pip install chemspipy
pip install mahalanobis
pip install lightgbm
pip install keras_tuner
pip install tensorflow
pip install os
pip install pathlib

Using trainsklearn.py... a list of 978 landmark genes is being used
Split into 80-10-10 - 10 for validation (hyperparameter tuning), 