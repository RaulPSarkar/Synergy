
import pandas as pd
import warnings
import sys
sys.path.append("..")
from src.drugToSmiles import drugToSMILES
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

##########################
##########################




pancreas = pd.read_csv(pancreasPath)
colon = pd.read_csv(colonPath)
breast = pd.read_csv(breastPath)
full = pd.concat([pancreas,colon, breast], axis=0)
#JOIN ALL DATASETS



drugNames = full.groupby(['Library Name'])['Synergy?'].count()



smilesTable = pd.DataFrame(columns=['drug', 'SMILES_A'])

for name in range(drugNames.shape[0]):
    smiles = drugToSMILES( drugNames.index[name], True, drugCached)
    if(smiles==-1):
        print(drugNames.index[name])
    else:
        smilesTable = smilesTable.append(pd.DataFrame([[drugNames.index[name], smiles]], columns=smilesTable.columns))

##CREATES A TABLE WITH ONLY THE DRUGS FOR WHICH SMILES ARE KNOWN (ALL BUT ONE IN THIS CASE)



out = full.merge(smilesTable, left_on='Library Name', right_on='drug')
smilesTable.columns = ['drug', 'SMILES_B']

out2 = out.merge(smilesTable, left_on='Anchor Name', right_on='drug')







out2.rename(columns={'Library Name': 'NSC1', 'Anchor Name': 'NSC2', 'SDIM': 'CELLNAME'}, inplace=True)

out2['GROUP'] = out2['NSC1'] + '_' + out2['NSC2']
#out2.to_csv('dataset.csv')



crispr = pd.read_csv('transcriptomics.csv', index_col=0)
crispr = crispr.fillna(0)
crispr.columns.name = 'CELLNAME'
crisprT = crispr.T

#crisprT = pd.read_csv('crisprT.csv')
#crispr.T.to_csv('crisprT.csv')

crispr.T.to_csv('transcriptomicsT.csv')


print("here")


df = crisprT.merge(out2, on=['CELLNAME'])

df = df[['Tissue', 'CELLNAME', 'NSC1', 'NSC2', 'Anchor Conc', 'SMILES_A', 'SMILES_B', 'GROUP', 'Delta Xmid', 'Delta Emax']]

#df.rename(columns={'Synergy?': 'COMBOSCORE'}, inplace=True)


#dgi = pd.read_csv('dgi.csv')
#columnNames = []

#for column in dgi.columns:
#    if column in crisprT.columns:
#        columnNames.append(column)
#crisprT = crisprT[columnNames]

#print(crisprT)
#crisprT.to_csv('crisprT.csv')

df.drop_duplicates(inplace=True)

df.to_csv('datasetMultiExpressionBreast2.csv')

