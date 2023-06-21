import numpy as np
import sys
import pandas as pd
import tensorflow as tf
import yaml
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
import keras_tuner
sys.path.append("..")
from src.buildDLModel import buildDL
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import os



##########################
##########################
#Default Params (for batch/direct run)
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
tunerTrials = 20 #how many trials the tuner will do for hyperparameter optimization
tunerRun = 2 #increase if you want to start the hyperparameter optimization process anew
kFold = 5 #number of folds to use for cross-validation
saveTopXHyperparametersPerFold = 3

tempFolder = Path(__file__).parent / 'tempFolder' / 'test.log'
##########################
##########################


data = pd.read_csv(data)
omics = pd.read_csv(omics, index_col=0)
fingerprints = pd.read_csv(fingerprints)
landmarkList = pd.read_csv(landmarkList,sep='\t')

landmarkList = landmarkList.loc[landmarkList['pr_is_lm'] == 1]

fullTunerDirectory = tunerDirectory / 'dl'
runString = 'run' + str(tunerRun)

def buildModel(hp):

    return buildDL(940, 1024, 
                   expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[256, 128]','[256, 128, 64]']),
                   drug_hlayers_sizes=hp.Choice('drug_hlayers_sizes',['[256, 128]','[256, 128, 64]']),
                   predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[256, 128]','[256, 128, 64]']),
                   hidden_activation=hp.Choice('hidden_activation',['relu','prelu']),
                   l2=hp.Choice('l2',[0.001, 0.0001]), 
                   hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4]),
                    learn_rate=hp.Choice('learn_rate', [0.001, 0.0001]))



def datasetToInput(data, omics, drugs):

    interceptionGenes = []
    for gene in landmarkList['pr_gene_symbol']:
        if gene in omics.T.columns:
            interceptionGenes.append(gene)

    omicsFinal = omics.T[  interceptionGenes  ]


    print("Generating Input Dataset. This may take a while...")
    setWithOmics = data.merge(omicsFinal, left_on='CELLNAME', right_index=True)
    print("Now merging with drug A...")
    setWithDrugA = setWithOmics.merge(drugs, on='SMILES_A')
    print("Now merging with drug B...")
    fullSet = setWithDrugA.merge(drugs, left_on='SMILES_B', right_on='SMILES_A')

    return fullSet







def datasetToFingerprint(data, drugs, smilesColumnName='SMILES_A'):

    indexes = data.index
    data = data[[smilesColumnName, 'Experiment']]
    fingerDF = data.merge(drugs, left_on=smilesColumnName, right_on='SMILES_A')
    fingerDF = fingerDF.reindex(indexes)

    #Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('SMILES')]
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('drug')]
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('Unnamed')]
    fingerDF = fingerDF.drop(['Experiment'], axis=1)

    return fingerDF



def datasetToOmics(data, omics):

    interceptionGenes = []
    for gene in landmarkList['pr_gene_symbol']:
        if gene in omics.T.columns:
            interceptionGenes.append(gene)

    omicsFinal = omics.T[  interceptionGenes  ]

    indexes = data.index
    data = data[['CELLNAME', 'Experiment','Delta Xmid', 'Delta Emax', 'Tissue', 'Anchor Conc', 'NSC1', 'NSC2']]
    setWithOmics = data.merge(omicsFinal, left_on='CELLNAME', right_index=True)
    setWithOmics = setWithOmics.reindex(indexes)
    
    supp = setWithOmics[['Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2', 'Experiment' ] ] 
    y = setWithOmics[['Delta Xmid', 'Delta Emax']]
    setWithOmics = setWithOmics.drop(['CELLNAME', 'Experiment', 'Delta Xmid', 'Delta Emax', 'Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2'], axis=1)

    #Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
    setWithOmics = setWithOmics.loc[:,~setWithOmics.columns.str.startswith('drug')]
    setWithOmics = setWithOmics.loc[:,~setWithOmics.columns.str.startswith('Unnamed')]

    return setWithOmics, y, supp





fullSet = datasetToInput(data,omics, fingerprints)

AfingerDF = datasetToFingerprint(data,fingerprints, 'SMILES_A')
BfingerDF = datasetToFingerprint(data,fingerprints, 'SMILES_B')
omicsDF, y, supp  = datasetToOmics(data, omics)


#print(AfingerDF['Experiment'])
#print(BfingerDF['Experiment'])
#print(omicsDF['Experiment'])
#print(supp)
#print(y)


kf = KFold(n_splits=kFold, shuffle=True)

fullPredictions = []
index = 0

superFinalHyperDF = []

for train_index , test_index in kf.split(y):
    suppTrain, suppTest = supp.iloc[train_index,:],supp.iloc[test_index,:]

    y_train , y_test = y.iloc[train_index, :] , y.iloc[test_index, :] #change if just 1 output var y[train_index]
    
    AfingerDFTrain, AfingerDFTest = AfingerDF.iloc[train_index,:],AfingerDF.iloc[test_index,:]
    BfingerDFTrain, BfingerDFTest = BfingerDF.iloc[train_index,:],BfingerDF.iloc[test_index,:]
    omicsDFTrain, omicsDFTest = omicsDF.iloc[train_index,:],omicsDF.iloc[test_index,:]

    #print(AfingerDFTrain)
    #print(BfingerDFTrain)
    #print(omicsDFTrain)


    XTrain = [omicsDFTrain, AfingerDFTrain, BfingerDFTrain]
    XTest = [omicsDFTest, AfingerDFTest, BfingerDFTest]


    runStringCV = runString + 'fold' + str(index)

    tuner = keras_tuner.BayesianOptimization(buildModel,objective='val_loss',max_trials=1, directory=fullTunerDirectory, project_name=runStringCV)
    tuner.search(XTrain, y_train, epochs=25, validation_data=(XTest, y_test))
    model = tuner.get_best_models()[0]

    ypred = np.squeeze(model.predict(XTest, batch_size=64))

    df = pd.DataFrame(data={'Experiment': suppTest['Experiment'],
                    'Cellname': suppTest['CELLNAME'],
                    'Library': suppTest['NSC1'],
                    'Anchor': suppTest['NSC2'],
                    'Tissue': suppTest['Tissue'],
                    'Conc': suppTest['Anchor Conc'],
                    'y_trueIC': y_test.iloc[:,0],
                    'y_trueEmax': y_test.iloc[:,1],
                    'y_predIC': ypred[:,0],
                    'y_predEmax': ypred[:,1]})

    path = outputPredictions / 'predictions.csv'

    saveTo = 'dl' + str(index) + '.csv'
    df.to_csv(outputPredictions / 'temp' / saveTo, index=False)
    fullPredictions.append(df)

    index += 1


outdir = outputPredictions / 'final' / 'dl'
if not os.path.exists(outdir):
    os.mkdir(outdir)


totalPreds = pd.concat(fullPredictions, axis=0)
finalName = 'dl' + runString + '.csv'
totalPreds.to_csv(outdir / finalName, index=False)
