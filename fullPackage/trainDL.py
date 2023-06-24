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
#fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
fingerprints = Path(__file__).parent / 'datasets/smiles2shuffledfingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
tunerTrials = 20 #how many trials the tuner will do for hyperparameter optimization
tunerRun = 8 #increase if you want to start the hyperparameter optimization process anew
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
                   expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   drug_hlayers_sizes=hp.Choice('drug_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   hidden_activation=hp.Choice('hidden_activation',['relu','prelu', 'leakyrelu']),
                   l2=hp.Choice('l2',[0.01, 0.001, 0.0001, 1e-05]), 
                   hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4,0.5]),
                    learn_rate=hp.Choice('learn_rate', [0.01, 0.001, 0.0001, 1e-05]))


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

    indexes = data['Experiment']
    data = data[[smilesColumnName, 'Experiment']]
    fingerDF = data.merge(drugs, left_on=smilesColumnName, right_on='SMILES_A')
    fingerDF = fingerDF.reindex(indexes)
    print(fingerDF)

    #Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('SMILES')]
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('drug')]
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('Unnamed')]
    fingerDF = fingerDF.drop(['Experiment','Experiment'], axis=1)

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
fullSet = fullSet.sample(frac=1)

#AfingerDF = datasetToFingerprint(data,fingerprints, 'SMILES_A')
#BfingerDF = datasetToFingerprint(data,fingerprints, 'SMILES_B')
#omicsDF, y, supp  = datasetToOmics(data, omics)

#print(supp)
#print(y)

#fullSet = datasetToInput(data,omics, fingerprints)

#supp is supplemental data (tissue type, id, etc, that will not be kept as an input)
supp = fullSet[ ['Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2', 'Experiment' ] ]

#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
X = X.loc[:,~X.columns.str.startswith('drug')]
X = X.loc[:,~X.columns.str.startswith('Unnamed')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis', 'Experiment'], axis=1)

y = fullSet[ ['Delta Xmid', 'Delta Emax' ] ]

ind = 0
omicsDF = X.iloc[:, 0: 940]
ind += 940
AfingerDF = X.iloc[:, ind: ind+1024]
ind += 1024
BfingerDF = X.iloc[:, ind: ind+1024]


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

    XTrain = [omicsDFTrain, AfingerDFTrain, BfingerDFTrain]
    XTest = [omicsDFTest, AfingerDFTest, BfingerDFTest]
    

    runStringCV = runString + 'fold' + str(index)

    #tuner = keras_tuner.BayesianOptimization(buildModel,objective='val_loss',max_trials=3, directory=fullTunerDirectory, project_name=runStringCV)
    #tuner.search(XTrain, y_train, epochs=20, validation_data=(XTest, y_test))


    tuner = keras_tuner.Hyperband(
    buildModel,
    max_epochs=30,
    objective='val_loss',
    executions_per_trial=1,
    directory=fullTunerDirectory,
    project_name=runStringCV
    )    


    tuner.search(x=XTrain,
             y=y_train,
             epochs=30,
             validation_split=0.2)


    #THIS PART WAS JUST COPIED FROM THE OFFICIAL KERAS DOCS
    #(After finding optimal HPs, fit the data) https://www.tensorflow.org/tutorials/keras/keras_tuner
    ######################################################
    ######################################################
    bestHP = tuner.get_best_hyperparameters(1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(bestHP)
    history = model.fit(XTrain, y_train, epochs=50, validation_split=0.2)

    valLossPerEpoch = history.history['val_loss']
    bestEpoch = valLossPerEpoch.index(max(valLossPerEpoch)) + 1
    print('Best epoch: %d' % (bestEpoch,))
    hypermodel = tuner.hypermodel.build(bestHP)
    # Retrain the model
    hypermodel.fit(XTrain, y_train, epochs=bestEpoch, validation_split=0.2)
    #####################################################
    ######################################################

    #model = tuner.get_best_models()[0]

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
