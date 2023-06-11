# Synergy

Running processGDSC.py will concatenate pancreas_anchor_combo.csv, colon_anchor_combo.csv, and breast_anchor_combo.csv, and generates a dataset called "dataset.csv" with an added SMILES A and SMILES B columns. It keeps only rows with known SMILES for both drugs (only 1 drug doesn't have a known SMILES).
A cached version of drug names to SMILES is present at drug2smiles.txt and is used for this process.

pip install pubchempy
pip install rdkit
pip install cirpy
pip install chemspipy