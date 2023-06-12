
import pandas as pd
import warnings
import sys
import numpy as np
sys.path.append("..")
from src.drugToSmiles import drugToSMILES, SMILEStoFingerprint
from src.mahalanobis import mahalanobisFunc
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
from pathlib import Path


##########################
##########################
#CHANGE PATHS AS NEEDED
pancreasPath = Path(__file__).parent / "datasets/pancreas_anchor_combo.csv.gz"
colonPath = Path(__file__).parent / "datasets/colon_anchor_combo.csv.gz"
breastPath = Path(__file__).parent / "datasets/breast_anchor_combo.csv.gz"
drugCached = Path(__file__).parent / "datasets/drug2smiles.txt"
omicsData = Path(__file__).parent / "datasets/crispr.csv.gz"
outputFile = Path(__file__).parent / "datasets/processedCRISPR.csv"
splitInto = 10 ##my code is bad, this is to lower RAM usage, leave as is
useCachedDrugs = True #keep as is, all drugs have been cached
##########################
##########################




def datasetToInput(data, omics, drugs):

    #print(omics.T)

    print("Generating Input Dataset. This may take a while...")
    setWithOmics = data.merge(omics.T, left_on='CELLNAME', right_index=True)
    print("Now merging with drugs...")
    setWithDrugs = setWithOmics.merge(drugs, left_on='', right_index=True)
    print(setWithDrugs)
    return data



pancreas = pd.read_csv(pancreasPath)
colon = pd.read_csv(colonPath)
breast = pd.read_csv(breastPath)
full = pd.concat([pancreas,colon, breast], axis=0)
#JOINS ALL DATASETS



drugNames = full.groupby(['Library Name'])['Synergy?'].count()



smilesTable = pd.DataFrame(columns=['drug', 'SMILES_A'])

for name in range(drugNames.shape[0]):
    smiles = drugToSMILES( drugNames.index[name], useCachedDrugs, drugCached)
    if(smiles==-1):
        print(drugNames.index[name])
    else:
        smilesTable = smilesTable.append(pd.DataFrame([[drugNames.index[name], smiles]], columns=smilesTable.columns))

##CREATES A TABLE WITH ONLY THE DRUGS FOR WHICH SMILES ARE KNOWN (ALL BUT ONE IN THIS CASE)
smilesTableBackup = smilesTable

SMILEStoFingerprint(smilesTableBackup)

out = full.merge(smilesTable, left_on='Library Name', right_on='drug')
smilesTable.columns = ['drug', 'SMILES_B']

out2 = out.merge(smilesTable, left_on='Anchor Name', right_on='drug')







out2.rename(columns={'Library Name': 'NSC1', 'Anchor Name': 'NSC2', 'SDIM': 'CELLNAME'}, inplace=True)

out2['GROUP'] = out2['NSC1'] + '_' + out2['NSC2']


crispr = pd.read_csv(omicsData, index_col=0)
crispr = crispr.fillna(0)
crispr.columns.name = 'CELLNAME'
crisprT = crispr.T
#crispr.T.to_csv('transcriptomicsT.csv')


#Otherwise there's not enough RAM
splitOut = np.array_split(out2, splitInto)
finalDF = []
for outDF in splitOut:
    print("here")
    df = crisprT.merge(outDF, on=['CELLNAME'])
    df = df[['Tissue', 'CELLNAME', 'NSC1', 'NSC2', 'Anchor Conc', 'SMILES_A', 'SMILES_B', 'GROUP', 'Delta Xmid', 'Delta Emax']]
    finalDF.append(df)


df = pd.concat(finalDF, axis=0)
df.drop_duplicates(inplace=True)

dfSuperFinal = mahalanobisFunc(df, ['Delta Xmid', 'Delta Emax'], ['CELLNAME', 'NSC1', 'NSC2'])

dfSuperFinal.to_csv(outputFile)

datasetToInput(dfSuperFinal, crispr, smilesTableBackup)
