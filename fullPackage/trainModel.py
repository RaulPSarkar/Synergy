import pandas as pd
import sys
import numpy as np
sys.path.append("..")

from pathlib import Path
from sklearn.model_selection import KFold, GroupShuffleSplit
import os
import keras_tuner
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import argparse
from sklearn.linear_model import Ridge
import tensorflow as tf
from src.buildDLModel import buildDL, buildNewDL

import matplotlib.pyplot as plt
from sklearn import tree


##########################
##########################

###########PARAMETERS
modelName = 'lgbm'  #en, rf, lgbm, svr, xgboost, base, ridge, dl, dlCoeffs, dlFull, dlMixed
crossValidationMode = 'regular' #drug, cell, regular
tunerTrials = 30 #how many trials the tuner will do for hyperparameter optimization
tunerRun = 7 #increase if you want to start the hyperparameter optimization process anew
kFold = 5 #number of folds to use for cross-validation
saveTopXHyperparametersPerFold = 3
useTopMutatedList = False
useCancerDrivers = True #whether to use cancer driver genes for coefficient branch
useSingleAgentResponse = False #adds single agent data  
useCoeffs = True #adds coefficient data to model. irrelevant if using a dl model
useDrugs = False #adds drug data to model. irrelevant if using a dl model
sensitivityAnalysisMode = False #whether to run the script for data size sensitivity analysis.
#doesn't work with DL models
sensitivitySizeFractions = [0.01, 0.03, 0.06, 0.125, 0.17, 0.25, 0.375, 0.5, 0.625, 0.75, 0.85, 1] #trains the model with each of
#the small fractions of the full dataset (WITH resampling), and saves each result

sizePrints = 1024


############FILEPATHS
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
#fingerprints = Path(__file__).parent / 'datasets/smiles2shuffledfingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
top3000MutatedList = Path(__file__).parent / 'datasets/top15mutatedgenes.tsv'
cancerDriverGenes = Path(__file__).parent / 'datasets/cancerDriverGenes.csv'
outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
#tunerDirectory = Path('W:\Media') / 'tuner'
coeffs = Path(__file__).parent / 'datasets/coefsProcessed.csv'


##########################
##########################






parser = argparse.ArgumentParser(description="Training synergy prediction models with sklearn")

parser.add_argument(
    "-m",
    "--model",
    default=modelName,
    help="Name of the model to train: en, rf, lgbm, svr, xgboost, base",
)
parser.add_argument(
    "-p",
    "--data",
    default=data,
    help="Processed combo file (from processGDSC.py)",
)

parser.add_argument(
    "-o",
    "--omics",
    default=omics,
    help="Omics Dataset file",
)

parser.add_argument(
    "-f",
    "--fingerprints",
    default=fingerprints,
    help="Smiles To Fingerprint file",
)

parser.add_argument(
    "-l",
    "--landmarkList",
    default=landmarkList,
    help="Landmark genes file",
)

parser.add_argument(
    "-output",
    "--predictions",
    default=outputPredictions,
    help="Output Predictions Base Directory",
)

parser.add_argument(
    "-trials",
    "--tunerTrials",
    default=tunerTrials,
    help="Number of Trials for each Fold for the tuning process",
)

parser.add_argument(
    "-t",
    "--tunerDirectory",
    default=tunerDirectory,
    help="Tuner Base Directory",
)


parser.add_argument(
    "-run",
    "--tunerRun",
    default=tunerRun,
    help="Tuner Run Number (Use the same number if wanting to use already computed trials)",
)

parser.add_argument(
    "-fold",
    "--kFold",
    default=kFold,
    help="Number of folds for cross validation",
)

parser.add_argument(
    "-hyper",
    "--saveTopXHyperparametersPerFold",
    default=saveTopXHyperparametersPerFold,
    help="Number of hyperparameters to store in a file adjacent to predictions (merely informative)",
)

parser.add_argument(
    "-validation",
    "--crossValidationMode",
    default=crossValidationMode,
    help="Style of cross validation: regular, cell, drug (leaves certain drug pairs/cell lines out for training)",
)

parser.add_argument(
    "-coeffdir",
    "--coeffs",
    default=coeffs,
    help="Path to the coefficients file",
)


args = parser.parse_args()

modelName = args.model
data = args.data
omics = args.omics
fingerprints = args.fingerprints
landmarkList = args.landmarkList
outputPredictions = args.predictions
tunerDirectory = args.tunerDirectory
tunerTrials = int(args.tunerTrials)
tunerRun = int(args.tunerRun)
kFold = args.kFold
saveTopXHyperparametersPerFold = args.saveTopXHyperparametersPerFold
crossValidationMode = args.crossValidationMode
coeffs = args.coeffs

top3000MutatedList = pd.read_csv(top3000MutatedList,sep='\t', index_col=0)
cancerDriverGenes = pd.read_csv(cancerDriverGenes)


#print(tf.config.list_physical_devices('GPU'))
print("Model selected: " + modelName)
if (modelName.strip()=='dl'):
    useCoeffs = False
    useDrugs = True
elif(modelName.strip()=='dlCoeffs'):
    useCoeffs = True
    useDrugs = False
elif(modelName.strip()=='dlFull'):
    useCoeffs = True
    useDrugs = True
elif(modelName.strip()=='dlMixed'):
    useCoeffs = True
    #useDrugs = False
    #why not?

if not os.path.exists(outputPredictions):
    os.mkdir(outputPredictions)



#THE TUNER IS ALREADY USING CV AUTOMATICALLY, SO I ONLY HAD TO DO ONE TRAIN-TEST SPLIT
#(FOR NESTED CV)

data = pd.read_csv(data)
omics = pd.read_csv(omics, index_col=0)
fingerprints = pd.read_csv(fingerprints)
landmarkList = pd.read_csv(landmarkList,sep='\t')
coeffs = pd.read_csv(coeffs, index_col=0)

landmarkList = landmarkList.loc[landmarkList['pr_is_lm'] == 1]


def buildModel(hp):
    use = modelName

    if(use=='dlFull'):
        model = buildDL(expr_dim = sizeOmics, 
                drug_dim = sizePrints,
                coeffs_dim= sizeCoeffs,
                useCoeffs=True,
                useDrugs=True,
                useSingleAgent=useSingleAgentResponse,
                expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                drug_hlayers_sizes=hp.Choice('drug_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                coeffs_hlayers_sizes=hp.Choice('coeffs_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                hidden_activation=hp.Choice('hidden_activation',['relu','prelu', 'leakyrelu']),
                l2=hp.Choice('l2',[0.01, 0.001, 0.0001, 1e-05]), 
                hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4,0.5]),
                learn_rate=hp.Choice('learn_rate', [0.01, 0.001, 0.0001, 1e-05]))
    

    elif(use=='dlMixed'):
        model = buildDL(expr_dim = sizeOmics, 
                drug_dim = sizePrints,
                useSingleAgent=useSingleAgentResponse,
                mixedModel=True,
                drug_hlayers_sizes=hp.Choice('drug_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                hidden_activation=hp.Choice('hidden_activation',['relu','prelu', 'leakyrelu']),
                l2=hp.Choice('l2',[0.01, 0.001, 0.0001, 1e-05]), 
                hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4,0.5]),
                learn_rate=hp.Choice('learn_rate', [0.01, 0.001, 0.0001, 1e-05]))


    elif(use=='dl'):
        model = buildDL(expr_dim = sizeOmics, 
                drug_dim = sizePrints,
                useCoeffs=False,
                useDrugs=True,
                useSingleAgent=useSingleAgentResponse,
                expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                drug_hlayers_sizes=hp.Choice('drug_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                hidden_activation=hp.Choice('hidden_activation',['relu','prelu', 'leakyrelu']),
                l2=hp.Choice('l2',[0.01, 0.001, 0.0001, 1e-05]), 
                hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4,0.5]),
                learn_rate=hp.Choice('learn_rate', [0.01, 0.001, 0.0001, 1e-05]))
    
    elif(use=='dlCoeffs'):
        model = buildDL(expr_dim = sizeOmics, 
                coeffs_dim= sizeCoeffs,
                useCoeffs=True,
                useDrugs=False,
                useSingleAgent=useSingleAgentResponse,
                expr_hlayers_sizes=hp.Choice('expr_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                coeffs_hlayers_sizes=hp.Choice('coeffs_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                predictor_hlayers_sizes=hp.Choice('predictor_hlayers_sizes',['[32]','[64,32]','[64]','[64, 64]','[64, 64, 64]','[256]','[256,256]','[128]','[128, 64]','[128, 64,32] ','[128, 128, 128]','[256, 128]','[256, 128, 64]','[512]','[1024, 512]','[1024, 512, 256]','[2048, 1024]']),
                hidden_activation=hp.Choice('hidden_activation',['relu','prelu', 'leakyrelu']),
                l2=hp.Choice('l2',[0.01, 0.001, 0.0001, 1e-05]), 
                hidden_dropout=hp.Choice('hidden_dropout', [0.1, 0.2,0.3,0.4,0.5]),
                learn_rate=hp.Choice('learn_rate', [0.01, 0.001, 0.0001, 1e-05]))



    elif(use=='rf'):
        model = ensemble.RandomForestRegressor(
            n_estimators=hp.Int('n_estimators', 100, 1000),
            max_depth=hp.Int('max_depth', 3, 55),
            max_features=hp.Choice('max_features', ['sqrt', 'log2']),
            min_samples_split=hp.Int('min_samples_split', 2, 5),
            min_samples_leaf=hp.Int('min_samples_leaf', 1, 5),
            bootstrap=hp.Boolean('bootstrap', True, False),
            #criterion=hp.Choice('criterion', ['gini','entropy']),
            n_jobs=-1
        )

    elif(use=='en'):
        model = MultiOutputRegressor ( ElasticNet(
            alpha=hp.Float('alpha', 0.1,10,  sampling="log"),
            l1_ratio=hp.Float('l1_ratio', 0.05, 1),
            max_iter=100000
        ) )
    elif(use=='svr'):
        model = MultiOutputRegressor ( LinearSVR(
            C = hp.Float('C', 1e-3, 1e3, sampling="log"), #1e-3
            epsilon = hp.Float('epsilon', 1e-4, 1e1, sampling="log"),
            dual=False,
            loss='squared_epsilon_insensitive',
            max_iter=100000
        ) )        

    elif(use=='lgbm'):
        #hyperparameters taken from https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        model = MultiOutputRegressor ( LGBMRegressor(
            #n_estimators=hp.Int('n_estimators', 100, 1000),
            n_estimators=hp.Int('n_estimators', 100, 3000),
            #learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling="log"), #1e-3
            learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling="log"), #1e-3
            #max_depth = hp.Int('max_depth', 3, 9),
            max_depth = hp.Int('max_depth', 3, 11),
            min_child_weight = hp.Int('min_child_weight', 1, 5),
            min_split_gain = hp.Float('min_split_gain', 0, 2), #1e-3
            subsample = hp.Float('subsample', 0.6, 1.0), #1e-3
            subsample_freq = hp.Choice('subsample_freq', ['0','1', '5'])
        ) )  

    elif(use=='xgboost'):
        model = MultiOutputRegressor (XGBRegressor(
            n_jobs=-1,
            n_estimators=hp.Int('n_estimators', 100, 1000),
            learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling="log"),
            max_depth = hp.Int('max_depth', 3, 9),
            min_child_weight = hp.Int('min_child_weight', 1, 5),
            gamma = hp.Float('gamma', 0, 2),
            subsample = hp.Float('subsample', 0.6, 1.0)
        ))
    elif(use=='ridge'):
        model = Ridge(
            alpha=hp.Float('alpha', 0.1,10,  sampling="log"),
        )

    return model




def datasetToInput(data, omics, drugs, coeffs):

    interceptionGenes = []
    interceptionCoeffs = []


    for gene in landmarkList['pr_gene_symbol']:
        if gene in omics.T.columns:
            interceptionGenes.append(gene)
            if( (not useTopMutatedList) and (not useCancerDrivers)):
                if gene in coeffs.index:
                    interceptionCoeffs.append(gene)
    
    if(useTopMutatedList):
        for gene in top3000MutatedList['Gene']:
            if gene in coeffs.index:
                interceptionCoeffs.append(gene)
    
    count = 0
    if(useCancerDrivers):
        
        for gene in cancerDriverGenes['symbol']:
            if gene in coeffs.index:
                count+= 1
                interceptionCoeffs.append(gene)

    print(count)
    


    omicsFinal = omics.T[  interceptionGenes  ]
    coeffsFinal = coeffs.T[interceptionCoeffs]
    coeffsFinal['drug'] = coeffsFinal.index
    coeffsFinal['drug'] = coeffsFinal['drug']
    coeffsFinal= coeffsFinal.fillna(0)

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

    if(useCoeffs and useDrugs):
        print("Now merging with coeffs A...")
        setWithDrugA = setWithOmics.merge(coeffsFinalA, left_on='NSC1', right_on='drugA')
        print("Now merging with coeffs B...")
        fullSet = setWithDrugA.merge(coeffsFinalB, left_on='NSC2', right_on='drugB')
        print("Now merging with drug A fingerprint...")
        fullSetA = fullSet.merge(drugs, on='SMILES_A')
        print("Now merging with drug B fingerprint...")
        fullSetB = fullSetA.merge(drugs, left_on='SMILES_B', right_on='SMILES_A')
    elif(useCoeffs):
        print("Now merging with coeffs A...")
        setWithDrugA = setWithOmics.merge(coeffsFinalA, left_on='NSC1', right_on='drugA')
        print("Now merging with coeffs B...")
        fullSetB = setWithDrugA.merge(coeffsFinalB, left_on='NSC2', right_on='drugB')
    elif(useDrugs):
        print("Now merging with drug A fingerprint...")
        fullSetA = setWithOmics.merge(drugs, on='SMILES_A')
        print("Now merging with drug B fingerprint...")
        fullSetB = fullSetA.merge(drugs, left_on='SMILES_B', right_on='SMILES_A')


    return fullSetB, len(interceptionGenes), len(interceptionCoeffs)



fullSet, sizeOmics, sizeCoeffs = datasetToInput(data,omics, fingerprints, coeffs)
fullSet = fullSet.sample(frac=1)
groupDrugs = fullSet['NSC1'] + fullSet['NSC2']
groupCell = fullSet['CELLNAME']

#supp is supplemental data (tissue type, id, etc, that will not be kept as an input)
supp = fullSet[ ['Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2', 'Experiment' ] ]

#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
X = X.loc[:,~X.columns.str.startswith('drug')]
X = X.loc[:,~X.columns.str.startswith('Unnamed')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis', 'Experiment'], axis=1)

singleAgentDF = X.loc[:, ['Library IC50','Library Emax']]

if(not useSingleAgentResponse or  (modelName=='dl' or modelName=='dlCoeffs' or modelName=='dlFull' or modelName=='dlMixed')   ):
    X = X.drop(['Library IC50','Library Emax'], axis=1)
    #I'm deleting these columns case it's a DL model, to make selecting each DF from X easier up ahead.
    #This is why I created singleAgentDF earlier
    

y = fullSet[ ['Delta Xmid', 'Delta Emax' ] ]
#make this a function maybe

#print(X)
#print(X.columns.to_list()) - Just to make sure everything was okay, which it was

#hyperparam tuning
runString = 'run' + str(tunerRun)




if(modelName=='dl' or modelName=='dlCoeffs' or modelName=='dlFull' or modelName=='dlMixed'):
    ind = 0
    omicsDF = X.iloc[:, ind: ind+sizeOmics]
    ind += sizeOmics

    if(useCoeffs):
        AcoeffsDF = X.iloc[:, ind: ind+sizeCoeffs]
        ind += sizeCoeffs
        BcoeffsDF = X.iloc[:, ind: ind+sizeCoeffs]
        ind += sizeCoeffs

    if(useDrugs):
        AfingerDF = X.iloc[:, ind: ind+sizePrints]
        ind += sizePrints
        BfingerDF = X.iloc[:, ind: ind+sizePrints]

if(modelName=='dlMixed'):
    
    omicsDFB = omicsDF.add_suffix('Blist')
    omicsDFA = omicsDF.add_suffix('Alist')

    #i am so goddamn lazy, don't copy paste like this
    interceptionGenes = []
    for gene in omicsDFA.columns:
        if gene in AcoeffsDF.columns:
            interceptionGenes.append(gene)

    #this is even worse, but technically it works
    interceptionGenesB = []
    
    for gene in omicsDFB.columns:
        if gene in BcoeffsDF.columns:
            interceptionGenesB.append(gene)

    omicsDF=omicsDFA[interceptionGenes]
    AcoeffsDF=AcoeffsDF[interceptionGenes]
    BcoeffsDF=BcoeffsDF[interceptionGenesB]


    AcoeffsDF = 1 - AcoeffsDF
    BcoeffsDF = 1 - BcoeffsDF

    AcoeffsDF.columns = omicsDF.columns
    BcoeffsDF.columns = omicsDF.columns

    omicsModifiedInput = omicsDF * AcoeffsDF * BcoeffsDF
    sizeOmics = len(omicsModifiedInput.columns)


def trainTestModel(sens=False, sensRun=0):

        #cross validation
    if(crossValidationMode=='drug'):
        gs = GroupShuffleSplit(n_splits=kFold)
        splits = gs.split(X, y, groupDrugs)
    elif(crossValidationMode=='cell'):
        gs = GroupShuffleSplit(n_splits=kFold)
        splits = gs.split(X, y, groupCell)
    else:
        kf = KFold(n_splits=kFold, shuffle=True)
        splits = kf.split(X)

    fullPredictions = []
    index = 0

    superFinalHyperDF = []


    for train_index , test_index in splits:
        if(sens and index>0):
            break
        #just do a single fold for sensitivity analysis

        suppTrain, suppTest = supp.iloc[train_index,:],supp.iloc[test_index,:]
        y_train , y_test = y.iloc[train_index, :] , y.iloc[test_index, :] #change if just 1 output var y[train_index]

        if(modelName!='dl' and modelName!='dlCoeffs' and modelName!='dlFull' and modelName!='dlMixed'):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        else:

            if(modelName!='dlMixed'):

                omicsDFTrain, omicsDFTest = omicsDF.iloc[train_index,:],omicsDF.iloc[test_index,:]

                if(useCoeffs):
                    AcoeffsDFTrain, AcoeffsDFTest = AcoeffsDF.iloc[train_index,:],AcoeffsDF.iloc[test_index,:]
                    BcoeffsDFTrain, BcoeffsDFTest = BcoeffsDF.iloc[train_index,:],BcoeffsDF.iloc[test_index,:]

                if(useDrugs):
                    AfingerDFTrain, AfingerDFTest = AfingerDF.iloc[train_index,:],AfingerDF.iloc[test_index,:]
                    BfingerDFTrain, BfingerDFTest = BfingerDF.iloc[train_index,:],BfingerDF.iloc[test_index,:]
                if(useSingleAgentResponse):
                    singleAgentDFTrain, singleAgentDFTest = singleAgentDF.iloc[train_index,:],singleAgentDF.iloc[test_index,:]

                XTrain = [omicsDFTrain]
                XTest = [omicsDFTest]

                if(useCoeffs):
                    XTrain.append(AcoeffsDFTrain)
                    XTrain.append(BcoeffsDFTrain)
                    XTest.append(AcoeffsDFTest)
                    XTest.append(BcoeffsDFTest)
                if(useDrugs):
                    XTrain.append(AfingerDFTrain)
                    XTrain.append(BfingerDFTrain)
                    XTest.append(AfingerDFTest)
                    XTest.append(BfingerDFTest)
                if(useSingleAgentResponse):
                    XTrain.append(singleAgentDFTrain)
                    XTest.append(singleAgentDFTest)

            else:
                singleAgentDFTrain, singleAgentDFTest = singleAgentDF.iloc[train_index,:],singleAgentDF.iloc[test_index,:]
                omicsDFTrain, omicsDFTest = omicsModifiedInput.iloc[train_index,:],omicsModifiedInput.iloc[test_index,:]
                if(useDrugs):
                    AfingerDFTrain, AfingerDFTest = AfingerDF.iloc[train_index,:],AfingerDF.iloc[test_index,:]
                    BfingerDFTrain, BfingerDFTest = BfingerDF.iloc[train_index,:],BfingerDF.iloc[test_index,:]

                XTrain = [omicsDFTrain]
                XTest = [omicsDFTest]

                if(useDrugs):
                    XTrain.append(AfingerDFTrain)
                    XTrain.append(BfingerDFTrain)
                    XTest.append(AfingerDFTest)
                    XTest.append(BfingerDFTest)

                XTrain.append(singleAgentDFTrain)
                XTest.append(singleAgentDFTest)

                

        if(modelName!='base'):
        
            fullTunerDirectory = tunerDirectory / modelName

            runStringCV = runString + 'fold' + str(index)
            if(sens):
                runStringCV = runString + 'size' + str(sensRun)


            hyperList = []

            if(modelName!='dl' and modelName!='dlCoeffs' and modelName!='dlFull' and modelName!='dlMixed'):
                tuner = keras_tuner.tuners.SklearnTuner(
                    oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                        objective=keras_tuner.Objective('score', 'min'),
                        max_trials=tunerTrials),
                    hypermodel=buildModel,
                    scoring=metrics.make_scorer(metrics.mean_squared_error),
                    cv=model_selection.KFold(5),
                    directory= fullTunerDirectory,
                    project_name= runStringCV)


                
                tuner.search(X_train, y_train.to_numpy())
                best_hp = tuner.get_best_hyperparameters()[0]
                #technically, no need to grab best HPs anymore because it's already fitted to training dataset
                #however, it's not a fully trained model on all the data: https://keras.io/api/keras_tuner/tuners/base_tuner/
            else:
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



            

            for hyper in range (saveTopXHyperparametersPerFold):
                best_hpIter = tuner.get_best_hyperparameters(saveTopXHyperparametersPerFold)[hyper]
                bestValsDict = best_hpIter.values
                bestValsDict = {k:[v] for k,v in bestValsDict.items()} #taken from https://stackoverflow.com/questions/57631895/dictionary-to-dataframe-error-if-using-all-scalar-values-you-must-pass-an-ind
                bestHyperDF = pd.DataFrame.from_dict(bestValsDict)
                bestHyperDF['fold'] = index+1
                bestHyperDF['rank'] = hyper+1
                hyperList.append(bestHyperDF)
            
            finalHyperDF = pd.concat(hyperList, axis=0)
            superFinalHyperDF.append(finalHyperDF)

        
        if(modelName!='base'):
            if(modelName!='dl' and modelName!='dlCoeffs' and modelName!='dlFull' and modelName!='dlMixed'):
                model = buildModel(best_hp)
                model.fit(X_train, y_train)
                #print(model.estimators_[0].coef_ )
                ypred = model.predict(X_test)

                #copied from https://stackoverflow.com/questions/40155128/plot-trees-for-a-random-forest-in-python-with-scikit-learn
                #if(modelName=='rf'):
                #    for i in range(3):
                #        fn=X.columns
                #        cn=y.columns
                #        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (40,40), dpi=1600)
                #        tree.plot_tree(model.estimators_[0],
                #                    feature_names = fn, 
                #                    class_names=cn,
                #                    filled = True,
                #                    fontsize=10)
                #        name = 'rfNew' + str(i) + '.png'
                #        fig.savefig(name)


            else:
                #THIS PART WAS JUST COPIED FROM THE OFFICIAL KERAS DOCS
                #(After tuning optimal HPs, fit the data) 
                #https://www.tensorflow.org/tutorials/keras/keras_tuner
                ######################################################
                ######################################################
                bestHP = tuner.get_best_hyperparameters(1)[0]

                # Build the model with the optimal hyperparameters and train it on the data for 65 epochs
                model = tuner.hypermodel.build(bestHP)
                history = model.fit(XTrain, y_train, epochs=65, validation_split=0.2)

                valLossPerEpoch = history.history['val_loss']
                bestEpoch = valLossPerEpoch.index(min(valLossPerEpoch)) + 1
                print('Best epoch: %d' % (bestEpoch,))
                hypermodel = tuner.hypermodel.build(bestHP)
                # Retrain the model -> i could just save the model instead maybe?
                hypermodel.fit(XTrain, y_train, epochs=bestEpoch, validation_split=0.2)
                #####################################################
                ######################################################
                ypred = np.squeeze(hypermodel.predict(XTest, batch_size=64))



            df = pd.DataFrame(data={'Experiment': suppTest['Experiment'].values,
                            'Cellname': suppTest['CELLNAME'].values,
                            'Library': suppTest['NSC1'].values,
                            'Anchor': suppTest['NSC2'].values,
                            'Tissue': suppTest['Tissue'].values,
                            'Conc': suppTest['Anchor Conc'].values,
                            'y_trueIC': y_test.iloc[:,0].values,
                            'y_trueEmax': y_test.iloc[:,1].values,
                            'y_predIC': ypred[:,0],
                            'y_predEmax': ypred[:,1]})

        else:
            dataTrain = data.iloc[train_index,:]
            dataTest = data.iloc[test_index,:]
            meanScores = dataTrain.groupby(['NSC1', 'NSC2'])['Delta Xmid', 'Delta Emax'].mean()

            predictedTest = dataTest.merge(meanScores, on=['NSC1', 'NSC2'])

            df = pd.DataFrame(data={
                                'Experiment': predictedTest['Experiment'],
                                'Cellname': predictedTest['CELLNAME'],
                                'Library': predictedTest['NSC1'],
                                'Anchor': predictedTest['NSC2'],
                                'Tissue': predictedTest['Tissue'],
                                'Conc': predictedTest['Anchor Conc'],
                                'y_trueIC': predictedTest['Delta Xmid_x'],
                                'y_predIC': predictedTest['Delta Xmid_y'],
                                'y_trueEmax': predictedTest['Delta Emax_x'],
                                'y_predEmax': predictedTest['Delta Emax_y']

                                })


        saveTo = modelName + str(index) + '.csv'
        df.to_csv(outputPredictions / 'temp' / saveTo, index=False)
        index+=1
        fullPredictions.append(df)

    emptyString = ''
    if(useSingleAgentResponse):
        emptyString = 'plusSingle'

    if(useCoeffs):
        emptyString += 'plusCoeffs'
    if(useDrugs):
        emptyString += 'plusDrugs'


    totalPreds = pd.concat(fullPredictions, axis=0)

    finalName = modelName + runString + crossValidationMode + emptyString + '.csv'
    finalHPName = modelName + runString + 'hyperParams.csv'

    if(sens): 
        finalName = modelName + runString + str(sensRun) + crossValidationMode + emptyString + '.csv'
        finalHPName = modelName + runString + str(sensRun) + 'hyperParams.csv'


    outdir = outputPredictions / 'final' / modelName
    if not os.path.exists(outdir):
        os.mkdir(outdir)



    totalPreds.to_csv(outdir / finalName, index=False)
    superFinalHyperDF = pd.concat(superFinalHyperDF, axis=0)
    superFinalHyperDF.to_csv(outdir / finalHPName, index=False)

    print("Best HP values:")
    print(superFinalHyperDF)

if(sensitivityAnalysisMode):
    originalX = X
    originalY = y
    ind = 0
    fullDF = pd.concat([originalY,originalX], axis=1)

    for sampleSize in sensitivitySizeFractions:
        sampledDF = fullDF.sample(frac=sampleSize, random_state=1)
        y = sampledDF.iloc[:, :2]
        X = sampledDF.iloc[:, 2:]
        print(y)
        print(X)
        #X = originalX.sample(frac=sampleSize, random_state=1) 
        #y = originalY.sample(frac=sampleSize, random_state=1)
        trainTestModel(sens=True, sensRun=ind)
        ind += 1
else:
    trainTestModel()