
import pandas as pd
import warnings
import sys
import numpy as np
from sklearn.utils import shuffle
sys.path.append("..")
from src.drugToSmiles import drugToSMILES, SMILEStoFingerprint
from src.mahalanobis import mahalanobisFunc
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

##########################
##########################
#CHANGE PATHS AS NEEDED
pancreasPath = Path(__file__).parent / "datasets/pancreas_anchor_combo.csv.gz"
colonPath = Path(__file__).parent / "datasets/colon_anchor_combo.csv.gz"
breastPath = Path(__file__).parent / "datasets/breast_anchor_combo.csv.gz"
drugCached = Path(__file__).parent / "datasets/drug2smiles.txt"
omicsData = Path(__file__).parent / "datasets/crispr.csv.gz"
singleAgent = Path(__file__).parent / "datasets/drugresponse.csv"
outputFile = Path(__file__).parent / "datasets/processedCRISPR.csv"
outputSMILEStoFingerprints = Path(__file__).parent / "datasets/smiles2fingerprints.csv"
outputSMILEStoShuffledFingerprints = Path(__file__).parent / "datasets/smiles2shuffledfingerprints.csv"
outputSMILEStoDummyFingerprints = Path(__file__).parent / "datasets/smiles2dummyfingerprints.csv"
splitInto = 10 ##my code is bad, this is to lower RAM usage, leave as is
useCachedDrugs = True #keep as is, all drugs have been cached
shuffleFingerprintBits = True #if activated, for each row, this will randomly shuffle the position of each fingerprint bit.
createDummifiedFingerprint = True #if activated, for each row, this will create a one-hot encoding of drugs, instead of creating fingerprints
#i.e., each drug will lose its chemical information, and get attributed completely random information (while keeping the number of bits)
##########################
##########################




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

fingerprintTable = SMILEStoFingerprint(smilesTableBackup)
fingerprintTable.to_csv(outputSMILEStoFingerprints, index=False)


allRows = []
if(shuffleFingerprintBits):
    for index, row in fingerprintTable.iterrows(): 
        
        row= row.iloc[0:1024]
        shuffledRow = shuffle(row, n_samples=len(row))
        shuffledRowDF = pd.DataFrame([shuffledRow.to_numpy()], columns=fingerprintTable.columns[0:1024])
        allRows.append(shuffledRowDF)

shuffledFingerprintTable = pd.concat(allRows, axis=0)
shuffledFingerprintTable['drug'] =fingerprintTable['drug']
shuffledFingerprintTable['SMILES_A'] =fingerprintTable['SMILES_A']
shuffledFingerprintTable.to_csv(outputSMILEStoShuffledFingerprints, index=False)


if(createDummifiedFingerprint):
    DFforDummy = fingerprintTable[['drug','SMILES_A']]
    print(DFforDummy['drug'])
    DFFingerDummy = pd.get_dummies(DFforDummy['drug'])

    #taken from https://stackoverflow.com/questions/42847441/renaming-columns-using-numbers-from-a-range-in-python-pandas
    DFFingerDummy = DFFingerDummy.rename(columns={x:y for x,y in zip(DFFingerDummy.columns,range(0,len(DFFingerDummy.columns)))})

    DFFingerDummy['drug'] =DFforDummy['drug']
    DFFingerDummy['SMILES_A'] =DFforDummy['SMILES_A']
    DFFingerDummy.to_csv(outputSMILEStoDummyFingerprints, index=False)



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


####GENERATE A DATAFRAME WITH SINGLE AGENT (IC50) VALUES
singleAgent = pd.read_csv(singleAgent)
singleAgent[['ID','Drug', 'GDSC']] = singleAgent['Unnamed: 0'].str.split(';',expand=True)
singleAgent.drop(['Unnamed: 0', 'ID', 'GDSC'], axis=1, inplace=True)
singleAgentSIDMCols = singleAgent.columns[:-1]
#singleAgent = singleAgent.drop_duplicates('Drug', keep='last')
plz = pd.melt(singleAgent, id_vars='Drug', value_vars=singleAgentSIDMCols)
plz = plz.dropna(how='any')
plz = plz.drop_duplicates(subset=['Drug','variable'], keep='last') #because its GDSC2
#df = pd.pivot(singleAgent, index='Drug', columns='SIDM00023', values='EN_coef')
#test = pd.wide_to_long(df=singleAgent, stubnames='SIDM', i='Drug', j='Combo') 
#print(test)
#THIS CODE IS COMPLETELY IRRELEVANT, I'LL DELETE LATER, IT WAS JUST CAUSE OTHER THING DIDN'T WORK
######################


#Otherwise there's not enough RAM
splitOut = np.array_split(out2, splitInto)
finalDF = []
for outDF in splitOut:
    print("+1")
    df = crisprT.merge(outDF, on=['CELLNAME'])
    df = df[['Tissue', 'CELLNAME', 'NSC1', 'NSC2', 'Anchor Conc', 'SMILES_A', 'SMILES_B', 'GROUP', 'Delta Xmid', 'Delta Emax', 'Library IC50','Library Emax']]
    finalDF.append(df)


df = pd.concat(finalDF, axis=0)


df.drop_duplicates(inplace=True)

#THIS DIDN'T WORK (TO ADD ANCHOR SINGLE AGENT VALUES) :(
#But now it's working :)
singleAgent = df.groupby(['NSC1', 'CELLNAME']).mean().reset_index()
singleAgent = singleAgent[['NSC1', 'CELLNAME', 'Library IC50', 'Library Emax']]
singleAgent.columns = ['Anch', 'CELLNAME', 'Anchor IC50', 'Anchor Emax']
df = pd.merge(df, singleAgent, left_on=['NSC2', 'CELLNAME'], right_on = ['Anch', 'CELLNAME'])
df.drop(['Anch'], axis=1, inplace=True)
#df.rename(columns={'NSC1_x': 'NSC1'}, inplace=True)
print(df)
print(df.columns)


scaler = MinMaxScaler()
scaler.fit(df[['Library IC50']])
df[['Library IC50']] = scaler.transform(df[['Library IC50']])
scaler.fit(df[['Library Emax']])
df[['Library Emax']] = scaler.transform(df[['Library Emax']])
scaler.fit(df[['Anchor IC50']])
df[['Anchor IC50']] = scaler.transform(df[['Anchor IC50']])
scaler.fit(df[['Anchor Emax']])
df[['Anchor Emax']] = scaler.transform(df[['Anchor Emax']])

#minmax scale the single agent values since they're kinda big

dfSuperFinal = mahalanobisFunc(df, ['Delta Xmid', 'Delta Emax'], ['CELLNAME', 'NSC1', 'NSC2'])



print(dfSuperFinal)



dfSuperFinal['Experiment'] = dfSuperFinal.index #this is used as index to reorganize later
dfSuperFinal = dfSuperFinal.sample(frac=1)



dfSuperFinal.to_csv(outputFile)


