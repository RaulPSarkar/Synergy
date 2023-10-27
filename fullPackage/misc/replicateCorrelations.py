import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error



pancreasPath = Path(__file__).parent / "../datasets/pancreas_anchor_combo.csv.gz"
colonPath = Path(__file__).parent / "../datasets/colon_anchor_combo.csv.gz"
breastPath = Path(__file__).parent / "../datasets/breast_anchor_combo.csv.gz"
cellLineData = Path(__file__).parent / "../datasets/cellLineData.csv" #used to add cancer subtype information


pancreas = pd.read_csv(pancreasPath)
colon = pd.read_csv(colonPath)
breast = pd.read_csv(breastPath)
full = pd.concat([pancreas,colon, breast], axis=0)


print(full.columns.to_list())
print(full.duplicated(keep=False, subset=['Cell Line name', 'SDIM', 'Anchor Name', 'Anchor Conc', 'Maxc', 'Library Name', 'Tissue']) )
#full['isDuplicate'] = full.duplicated(keep=False, subset=['Cell Line name', 'SDIM', 'Anchor Name', 'Maxc', 'Anchor Conc', 'Library Name', 'Tissue'])
#duplicatesOnlyDF = full.loc[full['isDuplicate']==True]

#); colon: 4 (HCT-15, HT-29, SK-CO-1, SW620); pancreas: 5 (KP-1N, KP-4, MZ1-PC, PA-TU-8988T, SUIT-2

replicateCellLines = full[full['Cell Line name'].isin(['AU565', 'BT-474', 'CAL-85-1', 'HCC1937', 'MFM-223'])]
print(replicateCellLines.columns)
replicateCellLines['isDuplicate'] = replicateCellLines.duplicated(keep=False, subset=['Cell Line name', 'SDIM', 'Tissue', 'Cancer Type', 'Anchor Name', 'Anchor Target', 'Anchor Pathway', 'Anchor Conc', 'Library Name','library Target', ' Library Pathway', 'Maxc'])

print(replicateCellLines['isDuplicate'].value_counts() )

duplicatesOnlyDF = replicateCellLines.loc[replicateCellLines['isDuplicate']==True]


duplicatesOnlyDF.to_csv('testDuplicates.csv')

print(duplicatesOnlyDF)