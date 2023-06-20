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

    data = data.drop(['CELLNAME'], axis=1)


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
X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
X = X.loc[:,~X.columns.str.startswith('drug')]
X = X.loc[:,~X.columns.str.startswith('Unnamed')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis', 'Experiment'], axis=1)

y = fullSet[ ['Delta Xmid', 'Delta Emax' ] ]
#make it a function


#940 landmark genes
model = buildDL(940, 1024, '[10]', '[10]', '[10]')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(x=[omicsDF, AfingerDF, BfingerDF], y=y_train, epochs=500, batch_size=32,
                            callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                                    CSVLogger(tempFolder)],
                            validation_data=(X_test, y_test), workers=6,
                            use_multiprocessing=False, validation_batch_size=64)#, class_weight=weights)





# best number of epochs
best_n_epochs = np.argmin(history.history['val_loss']) + 1
print('best n_epochs: %s' % best_n_epochs)
