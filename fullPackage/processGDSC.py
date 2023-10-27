
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
addSingleAgentData = True
pancreasPath = Path(__file__).parent / "datasets/pancreas_anchor_combo.csv.gz"
colonPath = Path(__file__).parent / "datasets/colon_anchor_combo.csv.gz"
breastPath = Path(__file__).parent / "datasets/breast_anchor_combo.csv.gz"
cellLineData = Path(__file__).parent / "datasets/cellLineData.csv" #used to add cancer subtype information
drugCached = Path(__file__).parent / "datasets/drug2smiles.csv"
omicsData = Path(__file__).parent / "datasets/crispr.csv.gz"
singleAgent = Path(__file__).parent / "datasets/drugresponse.csv"
outputFile = Path(__file__).parent / "datasets/processedCRISPRwithSingle.csv"
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
cellLineData = pd.read_csv(cellLineData)
print(cellLineData)
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







out2.rename(columns={'Library Name': 'NSC1', 'Anchor Name': 'NSC2', 'SDIM': 'CELLNAME'}, inplace=True) #this was just for consistency with ALMANAC dataset, not really necessary

out2['GROUP'] = out2['NSC1'] + '_' + out2['NSC2']


crispr = pd.read_csv(omicsData, index_col=0)
crispr = crispr.fillna(0)
crispr.columns.name = 'CELLNAME'
crisprT = crispr.T
#crispr.T.to_csv('transcriptomicsT.csv')






out2 = out2.merge(cellLineData, left_on=['CELLNAME'], right_on=['model_id']) 
#Otherwise there's not enough RAM
splitOut = np.array_split(out2, splitInto)
finalDF = []
for outDF in splitOut:
    print("+1")
    df = crisprT.merge(outDF, on=['CELLNAME'])
    df = df[['Tissue', 'CELLNAME', 'NSC1', 'NSC2', 'Anchor Conc', 'SMILES_A', 'SMILES_B', 'GROUP', 'Delta Xmid', 'Delta Emax', 'Library IC50','Library Emax', 'cancer_type_detail', 'gender', 'ethnicity', 'age_at_sampling']]
    finalDF.append(df)


df = pd.concat(finalDF, axis=0)


df.drop_duplicates(inplace=True)

scaler = MinMaxScaler()

if(addSingleAgentData):
    #THIS DIDN'T WORK (TO ADD ANCHOR SINGLE AGENT VALUES) :(
    #But now it's working :)
    singleAgent = df.groupby(['NSC1', 'CELLNAME']).mean().reset_index()
    singleAgent = singleAgent[['NSC1', 'CELLNAME', 'Library IC50', 'Library Emax']]
    singleAgent.columns = ['Anch', 'CELL', 'Anchor IC50', 'Anchor Emax']
    df = pd.merge(df, singleAgent, left_on=['NSC2', 'CELLNAME'], right_on = ['Anch', 'CELL'])
    df.drop(['Anch', 'CELL'], axis=1, inplace=True)
    #df.rename(columns={'NSC1_x': 'NSC1'}, inplace=True)
    scaler.fit(df[['Anchor IC50']])
    df[['Anchor IC50']] = scaler.transform(df[['Anchor IC50']])
    scaler.fit(df[['Anchor Emax']])
    df[['Anchor Emax']] = scaler.transform(df[['Anchor Emax']])
    #minmax scale the single agent values since they're kinda big

scaler.fit(df[['Library IC50']])
df[['Library IC50']] = scaler.transform(df[['Library IC50']])
scaler.fit(df[['Library Emax']])
df[['Library Emax']] = scaler.transform(df[['Library Emax']])


dfSuperFinal = mahalanobisFunc(df, ['Delta Xmid', 'Delta Emax'], ['CELLNAME', 'NSC1', 'NSC2'])




dfSuperFinal = dfSuperFinal.fillna(df['age_at_sampling'].mean()) #to replace NaN age values

dfSuperFinal['cancerTypeFactorized'] = pd.factorize(dfSuperFinal['cancer_type_detail'])[0]
dfSuperFinal['genderFactorized'] = pd.factorize(dfSuperFinal['gender'])[0]
dfSuperFinal['ethnicityFactorized'] = pd.factorize(dfSuperFinal['ethnicity'])[0]

scaler.fit(dfSuperFinal[['cancerTypeFactorized']])
dfSuperFinal[['cancerTypeFactorized']] = scaler.transform(dfSuperFinal[['cancerTypeFactorized']])
scaler.fit(dfSuperFinal[['ethnicityFactorized']])
dfSuperFinal[['ethnicityFactorized']] = scaler.transform(dfSuperFinal[['ethnicityFactorized']])
scaler.fit(dfSuperFinal[['age_at_sampling']])
dfSuperFinal[['ageScaled']] = scaler.transform(dfSuperFinal[['age_at_sampling']])

print(dfSuperFinal)



dfSuperFinal['Experiment'] = dfSuperFinal.index #this is used as index to reorganize later
dfSuperFinal = dfSuperFinal.sample(frac=1)




dfSuperFinal.to_csv(outputFile)


