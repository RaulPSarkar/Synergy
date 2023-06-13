import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 

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





##########################
##########################
#Change
modelName = 'lgbm' #en, rf, lgbm, svr, xgboost
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'
outputPredictions = Path(__file__).parent / 'predictions'
tunerDirectory = Path(__file__).parent / 'tuner'
tunerTrials = 1
kFold = 5
##########################
##########################

#check models: i think it's not updated

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


    print(omicsFinal)
    print("Generating Input Dataset. This may take a while...")
    setWithOmics = data.merge(omicsFinal, left_on='CELLNAME', right_index=True)
    print("Now merging with drug A...")
    setWithDrugA = setWithOmics.merge(drugs, on='SMILES_A')
    print("Now merging with drug B...")
    fullSet = setWithDrugA.merge(drugs, left_on='SMILES_B', right_on='SMILES_A')

    return fullSet




fullSet = datasetToInput(data,omics, fingerprints)


#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
X = X.loc[:,~X.columns.str.startswith('drug')]
X = X.loc[:,~X.columns.str.startswith('Unnamed')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis'], axis=1)

y = fullSet[ ['Delta Xmid', 'Delta Emax' ] ]


print(y)


#cross validation
kf = KFold(n_splits=kFold)

fullPredictions = []
index = 0
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y.iloc[train_index, :] , y.iloc[test_index, :] #change if just 1 output var y[train_index]
    



    tuner = keras_tuner.tuners.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimizationOracle(
            objective=keras_tuner.Objective('score', 'min'),
            max_trials=tunerTrials),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.mean_squared_error),
        cv=model_selection.KFold(5),
        directory= tunerDirectory,
        project_name=modelName + '/' + str(index))


    print('X:')
    print(X_train)
    print('y:')
    print(y_train)

    
    tuner.search(X_train, y_train.to_numpy())
    best_model = tuner.get_best_models(num_models=1)[0]


    ypred = best_model.predict(X_test)


    df = pd.DataFrame(data={#'experiment': test_dataset.get_row_ids(sep='+'),
                            'Tissue': fullSet['Tissue'],
                            'Conc': fullSet['Anchor Conc'],
                            'y_trueIC': y_test.iloc[:,0],
                            'y_trueEmax': y_test.iloc[:,1],
                            'y_predIC': ypred[:,0],
                            'y_predEmax': ypred[:,1]})


    saveTo = modelName + str(index) + '.csv'
    df.to_csv(outputPredictions / saveTo, index=False)
    index+=1
    fullPredictions.append(df)


totalPreds = pd.concat(fullPredictions, axis=0)
finalName = modelName + '.csv'
df.to_csv(outputPredictions / 'final' / finalName, index=False)


#leave one out validation