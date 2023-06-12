import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 



##########################
##########################
#Change
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
fingerprints = Path(__file__).parent / 'datasets/smiles2fingerprints.csv'
landmarkList = Path(__file__).parent / 'datasets/landmarkgenes.txt'

kFold = 5
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
            print(gene)
            interceptionGenes.append(gene)

    omicsFinal = omics.T[  interceptionGenes  ]


    print(omicsFinal)
    print("Generating Input Dataset. This may take a while...")
    setWithOmics = data.merge(omicsFinal, left_on='CELLNAME', right_index=True)
    print("Now merging with drug A...")
    setWithDrugA = setWithOmics.merge(drugs, on='SMILES_A')
    print("Now merging with drug B...")
    fullSet = setWithDrugA.merge(drugs, left_on='SMILES_B', right_on='SMILES_A')
    print(fullSet)

    return fullSet




fullSet = datasetToInput(data,omics, fingerprints)


#Taken from https://stackoverflow.com/questions/19071199/drop-columns-whose-name-contains-a-specific-string-from-pandas-dataframe because I'm lazy
X = fullSet.loc[:,~fullSet.columns.str.startswith('SMILES')]
X = X.loc[:,~fullX.columns.str.startswith('drug')]
X = X.drop(['Tissue','CELLNAME','NSC1','NSC2','Anchor Conc','GROUP','Delta Xmid','Delta Emax','mahalanobis'], axis=1)






#cross validation

kf = KFold(n_splits=kFold)
model = LogisticRegression(solver= 'liblinear')

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
    
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)








import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] ="2"
import argparse
from datetime import datetime

import dill as pickle
import yaml
import pickle

from dataset import MultiInputDataset
from src.utils.utils import save_evaluation_results, get_ml_algorithm, evaluate_ml_model
import keras_tuner
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from imblearn.under_sampling import RandomUnderSampler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor



def build_model(hp):
    use = 'lgbm'
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


    #'XGBRegressor': XGBRegressor(),
    #'LGBMRegressor': 


    return model



output_dir = os.path.join('../results/', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(output_dir)

# save settings to file in output_dir
with open(os.path.join(output_dir, 'settings_used.yml'), 'w') as outfile:
    yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)


tuner = keras_tuner.tuners.SklearnTuner(
    oracle=keras_tuner.oracles.BayesianOptimizationOracle(
        objective=keras_tuner.Objective('score', 'min'),
        max_trials=2),
    hypermodel=build_model,
    scoring=metrics.make_scorer(metrics.mean_squared_error),
    cv=model_selection.KFold(5),
    directory='/Tuner',
    project_name='MultiRFlgbm')



tuner.search(val_dataset.X, val_dataset.y)
best_model = tuner.get_best_models(num_models=1)[0]


ypred = best_model.predict(test_dataset.X)


df = pd.DataFrame(data={'experiment': test_dataset.get_row_ids(sep='+'),
                        'Tissue': test_dataset.response_dataset['Tissue'],
                        'Conc': test_dataset.response_dataset['Anchor Conc'],
                        'y_trueIC': test_dataset.y[:,0],
                        'y_trueEmax': test_dataset.y[:,1],
                        'y_predIC': ypred[:,0],
                        'y_predEmax': ypred[:,1]})

df.to_csv('../results/predictionsMultiCrisprRFmae4.csv', index=False)




#loop hyperparameters/train



#leave one out validation