import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 
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




##########################
##########################
#Change
modelName = 'xgboost' #en, rf, lgbm, svr, xgboost
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
tunerTrials = 50
tunerRun = 0 #increase if you want to start the hyperparameter optimization process anew
kFold = 5
useBaselineInstead = False #change to true to use baseline instead of the model from modelName
##########################
##########################


if(useBaselineInstead):
    modelName = 'baseline'
data = pd.read_csv(data)
omics = pd.read_csv(omics, index_col=0)
fingerprints = pd.read_csv(fingerprints)
landmarkList = pd.read_csv(landmarkList,sep='\t')

landmarkList = landmarkList.loc[landmarkList['pr_is_lm'] == 1]


def build_model(hp):
    use = modelName
    if(use=='rf'):
        model = ensemble.RandomForestRegressor(
            n_estimators=hp.Int('n_estimators', 10, 150, step=2),
            criterion='absolute_error',
            max_depth=hp.Int('max_depth', 3, 30),
            bootstrap=hp.Boolean('bootstrap', True, False),
            #criterion=hp.Choice('criterion', ['gini','entropy']),
            n_jobs=-1
        )
    elif(use=='en'):
        model = MultiOutputRegressor ( ElasticNet(
            alpha=hp.Float('alpha', 1e-3,1e3,  sampling="log"),
            l1_ratio=hp.Float('l1_ratio', 0.1, 0.9),
            max_iter=100000
        ) )
    elif(use=='svr'):
        model = MultiOutputRegressor ( LinearSVR(
            C = hp.Float('C', 1e-3, 1e3, step=8, sampling="log"), #1e-3
            epsilon = hp.Float('epsilon', 1e-4, 1e1, step=6, sampling="log"),
            dual=False,
            loss='squared_epsilon_insensitive',
            max_iter=100000
        ) )        

    elif(use=='lgbm'):
        #hyperparameters taken from https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        model = MultiOutputRegressor ( LGBMRegressor(
            n_estimators=hp.Int('n_estimators', 100, 1000),
            learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling="log"), #1e-3
            max_depth = hp.Int('max_depth', 3, 9),
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

    return model




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




fullSet = datasetToInput(data,omics, fingerprints)

remainingData, validationData = train_test_split(fullSet, test_size=0.1, shuffle=True)

supp = remainingData[ ['Tissue', 'Anchor Conc', 'CELLNAME', 'NSC1', 'NSC2' ] ]

#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
X = remainingData.loc[:,~remainingData.columns.str.startswith('SMILES')]
X = X.loc[:,~X.columns.str.startswith('drug')]
X = X.loc[:,~X.columns.str.startswith('Unnamed')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis'], axis=1)

y = remainingData[ ['Delta Xmid', 'Delta Emax' ] ]



#create validation X and y
Xval = validationData.loc[:,~validationData.columns.str.startswith('SMILES')]
Xval = Xval.loc[:,~Xval.columns.str.startswith('drug')]
Xval = Xval.loc[:,~Xval.columns.str.startswith('Unnamed')]
Xval = Xval.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis'], axis=1)

yVal = validationData[ ['Delta Xmid', 'Delta Emax' ] ]


#hyperparam tuning

runString = 'run' + str(tunerRun)

if(not useBaselineInstead):
    
    fullTunerDirectory = tunerDirectory / modelName

    tuner = keras_tuner.tuners.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimizationOracle(
            objective=keras_tuner.Objective('score', 'min'),
            max_trials=tunerTrials),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.mean_squared_error),
        cv=model_selection.KFold(5),
        directory= fullTunerDirectory,
        project_name=runString )


    tuner.search(Xval, yVal.to_numpy())
    best_hp = tuner.get_best_hyperparameters()[0]




#cross validation
kf = KFold(n_splits=kFold)

fullPredictions = []
index = 0



for train_index , test_index in kf.split(X):
    suppTrain, suppTest = supp.iloc[train_index,:],supp.iloc[test_index,:]
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y.iloc[train_index, :] , y.iloc[test_index, :] #change if just 1 output var y[train_index]
    
    
    if(not useBaselineInstead):
        model = build_model(best_hp)
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)
        df = pd.DataFrame(data={'Cellname': suppTest['CELLNAME'],
                        'Library': suppTest['NSC1'],
                        'Anchor': suppTest['NSC2'],
                        'Tissue': suppTest['Tissue'],
                        'Conc': suppTest['Anchor Conc'],
                        'y_trueIC': y_test.iloc[:,0],
                        'y_trueEmax': y_test.iloc[:,1],
                        'y_predIC': ypred[:,0],
                        'y_predEmax': ypred[:,1]})

    else:
        dataTrain = data.iloc[train_index,:]
        dataTest = data.iloc[test_index,:]
        meanScores = dataTrain.groupby(['NSC1', 'NSC2'])['Delta Xmid', 'Delta Emax'].mean()

        predictedTest = dataTest.merge(meanScores, on=['NSC1', 'NSC2'])

        df = pd.DataFrame(data={
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


totalPreds = pd.concat(fullPredictions, axis=0)
finalName = modelName + runString + '.csv'
outdir = outputPredictions / 'final' / modelName
if not os.path.exists(outdir):
    os.mkdir(outdir)


totalPreds.to_csv(outdir / finalName, index=False)


#leave one out validation