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
data = Path(__file__).parent / 'datasets/processedCRISPRTemp.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
#fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
coeffs = Path(__file__).parent / 'datasets/coefsProcessed.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
top3000MutatedList = Path(__file__).parent / 'datasets/top15mutatedgenes.tsv'

outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
tunerTrials = 20 #how many trials the tuner will do for hyperparameter optimization
tunerRun = 3 #increase if you want to start the hyperparameter optimization process anew
kFold = 5 #number of folds to use for cross-validation
saveTopXHyperparametersPerFold = 3

sizeOmics = 940
sizeCoeffs = 2877
sizePrints = 1024

tempFolder = Path(__file__).parent / 'tempFolder' / 'test.log'
##########################
##########################



useCoeffs = True
useDrugs = True

data = pd.read_csv(data)
omics = pd.read_csv(omics, index_col=0)
fingerprints = pd.read_csv(fingerprints)
landmarkList = pd.read_csv(landmarkList,sep='\t')
top3000MutatedList = pd.read_csv(top3000MutatedList,sep='\t', index_col=0)
coeffs = pd.read_csv(coeffs, index_col=0)


landmarkList = landmarkList.loc[landmarkList['pr_is_lm'] == 1]

fullTunerDirectory = tunerDirectory / 'dlNew'
runString = 'run' + str(tunerRun)

def buildModel(hp):


    return buildDL(expr_dim = sizeOmics, 
                   drug_dim = sizePrints,
                   coeffs_dim= sizeCoeffs,
                   useCoeffs=useCoeffs,
                   useDrugs=useDrugs,
                   expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   drug_hlayers_sizes=hp.Choice('drug_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   coeffs_hlayers_sizes=hp.Choice('coeffs_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                   hidden_activation=hp.Choice('hidden_activation',['relu','prelu', 'leakyrelu']),
                   l2=hp.Choice('l2',[0.01, 0.001, 0.0001, 1e-05]), 
                   hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4,0.5]),
                    learn_rate=hp.Choice('learn_rate', [0.01, 0.001, 0.0001, 1e-05]))


def datasetToInput(data, omics, coeffs, drugs):


    interceptionGenes = []
    interceptionCoeffs = []
    for gene in landmarkList['pr_gene_symbol']:
        if gene in omics.T.columns:
            interceptionGenes.append(gene)
        #if gene in coeffs.index:
        #   interceptionCoeffs.append(gene)
    
    for gene in top3000MutatedList['Gene']:
        if gene in coeffs.index:
            interceptionCoeffs.append(gene)
    print(len( interceptionCoeffs ) )

    omicsFinal = omics.T[  interceptionGenes  ]
    coeffsFinal = coeffs.T[interceptionCoeffs]
    coeffsFinal['drug'] = coeffsFinal.index
    coeffsFinal['drug'] = coeffsFinal['drug']
    coeffsFinal= coeffsFinal.fillna(0)
    #print(coeffsFinal)

    #get list of all used drugs
    listOfDrugs = data['NSC1']
    listOfDrugs = listOfDrugs.drop_duplicates().to_list()
    #print(len (listOfDrugs) )

    coeffsFinal = coeffsFinal[coeffsFinal['drug'].isin(listOfDrugs)]

    #this is just to distinguish which genes come from which drug (or if they come from the omics dataset)
    coeffsFinalA = coeffsFinal.add_suffix('Alist')
    coeffsFinalB = coeffsFinal.add_suffix('Blist')
    coeffsFinalA = coeffsFinalA.rename(columns={'drugAlist': 'drugA'})
    coeffsFinalB = coeffsFinalB.rename(columns={'drugBlist': 'drugB'})


    print("Generating Input Dataset. This may take a while...")
    setWithOmics = data.merge(omicsFinal, left_on='CELLNAME', right_index=True)
    print("Now merging with coeffs A...")
    setWithDrugA = setWithOmics.merge(coeffsFinalA, left_on='NSC1', right_on='drugA')
    print("Now merging with coeffs B...")
    fullSet = setWithDrugA.merge(coeffsFinalB, left_on='NSC2', right_on='drugB')

    print("Now merging with drug A fingerprint...")
    fullSetA = fullSet.merge(drugs, on='SMILES_A')
    print("Now merging with drug B fingerprint...")
    fullSetB = fullSetA.merge(drugs, left_on='SMILES_B', right_on='SMILES_A')

    return fullSetB






fullSet = datasetToInput(data,omics, coeffs, fingerprints)
fullSet = fullSet.sample(frac=1)


#supp is supplemental data (tissue type, id, etc, that will not be kept as an input)
supp = fullSet[ ['Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2', 'Experiment' ] ]

#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
X = X.loc[:,~X.columns.str.startswith('drug')]
X = X.loc[:,~X.columns.str.startswith('Unnamed')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis', 'Experiment'], axis=1)

y = fullSet[ ['Delta Xmid', 'Delta Emax' ] ]

ind = 0
omicsDF = X.iloc[:, ind: ind+sizeOmics]
ind += sizeOmics
AcoeffsDF = X.iloc[:, ind: ind+sizeCoeffs]
ind += sizeCoeffs
BcoeffsDF = X.iloc[:, ind: ind+sizeCoeffs]

ind += sizeCoeffs
AfingerDF = X.iloc[:, ind: ind+sizePrints]
ind += sizePrints
BfingerDF = X.iloc[:, ind: ind+sizePrints]


kf = KFold(n_splits=kFold, shuffle=True)

fullPredictions = []
index = 0

superFinalHyperDF = []

for train_index , test_index in kf.split(y):
    suppTrain, suppTest = supp.iloc[train_index,:],supp.iloc[test_index,:]

    y_train , y_test = y.iloc[train_index, :] , y.iloc[test_index, :] #change if just 1 output var y[train_index]
    
    AcoeffsDFTrain, AcoeffsDFTest = AcoeffsDF.iloc[train_index,:],AcoeffsDF.iloc[test_index,:]
    BcoeffsDFTrain, BcoeffsDFTest = BcoeffsDF.iloc[train_index,:],BcoeffsDF.iloc[test_index,:]

    AfingerDFTrain, AfingerDFTest = AfingerDF.iloc[train_index,:],AfingerDF.iloc[test_index,:]
    BfingerDFTrain, BfingerDFTest = BfingerDF.iloc[train_index,:],BfingerDF.iloc[test_index,:]

    print(AfingerDFTrain)
    print(BfingerDFTrain)
    print(AcoeffsDFTrain)
    print(BcoeffsDFTrain)

    omicsDFTrain, omicsDFTest = omicsDF.iloc[train_index,:],omicsDF.iloc[test_index,:]

    XTrain = [omicsDFTrain, AcoeffsDFTrain, BcoeffsDFTrain, AfingerDFTrain, BfingerDFTrain]
    XTest = [omicsDFTest, AcoeffsDFTest, BcoeffsDFTest, AfingerDFTest, BfingerDFTest]
    

    runStringCV = runString + 'fold' + str(index)



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
    #(After tuning optimal HPs, fit the data) https://www.tensorflow.org/tutorials/keras/keras_tuner
    ######################################################
    ######################################################
    bestHP = tuner.get_best_hyperparameters(1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(bestHP)
    history = model.fit(XTrain, y_train, epochs=60, validation_split=0.2)

    valLossPerEpoch = history.history['val_loss']
    bestEpoch = valLossPerEpoch.index(min(valLossPerEpoch)) + 1
    print('Best epoch: %d' % (bestEpoch,))
    hypermodel = tuner.hypermodel.build(bestHP)
    # Retrain the model -> i could just save the model instead maybe?
    hypermodel.fit(XTrain, y_train, epochs=bestEpoch, validation_split=0.2)
    #####################################################
    ######################################################

    #model = tuner.get_best_models()[0]

    ypred = np.squeeze(hypermodel.predict(XTest, batch_size=64))

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

    saveTo = 'dlNew' + str(index) + '.csv'
    df.to_csv(outputPredictions / 'temp' / saveTo, index=False)
    fullPredictions.append(df)

    index += 1


outdir = outputPredictions / 'final' / 'dl'
if not os.path.exists(outdir):
    os.mkdir(outdir)


totalPreds = pd.concat(fullPredictions, axis=0)
finalName = 'dlNew' + runString + '.csv'
totalPreds.to_csv(outdir / finalName, index=False)
