import numpy as np
import sys
import pandas as pd
import tensorflow as tf
import yaml
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
sys.path.append("..")
from src.buildDLModel import buildDL
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 



##########################
##########################
#Default Params (for batch/direct run)
modelName = 'en' #en, rf, lgbm, svr, xgboost, base
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
tunerTrials = 20 #how many trials the tuner will do for hyperparameter optimization
tunerRun = 1 #increase if you want to start the hyperparameter optimization process anew
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
    data = data[[smilesColumnName]]
    fingerDF = data.merge(drugs, left_on=smilesColumnName, right_on='SMILES_A')

    #Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('SMILES')]
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('drug')]
    fingerDF = fingerDF.loc[:,~fingerDF.columns.str.startswith('Unnamed')]

    return fingerDF



def datasetToOmics(data, omics):

    interceptionGenes = []
    for gene in landmarkList['pr_gene_symbol']:
        if gene in omics.T.columns:
            interceptionGenes.append(gene)

    omicsFinal = omics.T[  interceptionGenes  ]

    data = data[['CELLNAME']]
    setWithOmics = data.merge(omicsFinal, left_on='CELLNAME', right_index=True)

    setWithOmics = setWithOmics.drop(['CELLNAME'], axis=1)

    #Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
    setWithOmics = setWithOmics.loc[:,~setWithOmics.columns.str.startswith('drug')]
    setWithOmics = setWithOmics.loc[:,~setWithOmics.columns.str.startswith('Unnamed')]

    return setWithOmics





fullSet = datasetToInput(data,omics, fingerprints)

AfingerDF = datasetToFingerprint(data,fingerprints, 'SMILES_A')
BfingerDF = datasetToFingerprint(data,fingerprints, 'SMILES_B')
omicsDF = datasetToOmics(data, omics)
#only keep correct columns
print(AfingerDF)
print(BfingerDF)
print(omicsDF)

#supp is supplemental data (tissue type, id, etc, that will not be kept as an input)
supp = fullSet[ ['Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2', 'Experiment' ] ]

#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
#X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
#X = X.loc[:,~X.columns.str.startswith('drug')]
#X = X.loc[:,~X.columns.str.startswith('Unnamed')]
#X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis', 'Experiment'], axis=1)

y = fullSet[ ['Delta Xmid', 'Delta Emax' ] ]
#make it a function


#940 landmark genes

#'expr_hlayers_sizes': '[256, 128]', 'drug_hlayers_sizes': '[256]', 'predictor_hlayers_sizes': '[256, 128, 64]', 'hidden_dropout': 0.4, 'hidden_activation': 'relu', 'l2': 0.0001, 'learn_rate': 0.001, 

model = buildDL(940, 1024, '[256, 128]', '[256, 128]', '[256, 128, 64]', hidden_activation='relu', l1=0, l2=0, input_dropout=0, hidden_dropout=0, learn_rate=0.001)


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

    #XTrain = np.asarray(XTrain).astype('float32')

    XTest = [omicsDFTest, AfingerDFTest, BfingerDFTest]
    #XTest = np.asarray(XTest).astype('float32')


    history = model.fit(x=XTrain, y=y_train, epochs=500, batch_size=32,
                                callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                                        CSVLogger(tempFolder)],
                                validation_data=(XTest, y_test), workers=6,
                                use_multiprocessing=False, validation_batch_size=64)#, class_weight=weights)







    # best number of epochs
    best_n_epochs = np.argmin(history.history['val_loss']) + 1
    print('best n_epochs: %s' % best_n_epochs)



    index += 1


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
    df.to_csv(path, index=False)
