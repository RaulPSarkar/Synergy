import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path



##########################
##########################
#Regular Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DLrun1.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensemble.csv'


#Shuffled Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun33.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DLrun6.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun33.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun33.csv']
#predictionWeights = [1,1,1,1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleShuffled.csv'


#Drug CV Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'dlrun1drug.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99drugplusDrugs.csv']
#predictionWeights = [1,1,1,1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleDrug.csv'


#Cell CV Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'dlrun1cell.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99cellplusDrugs.csv']
#predictionWeights = [1,1,1,1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleCell.csv'



#Coeffs Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DLrun1.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensemble.csv'



##########################
##########################


#ID is Experiment column


predictionsDFList = []
predictionsDF = pd.DataFrame()

index = 0

for path in predictionPaths:
    df = pd.read_csv(path)
    df.sort_values(by='Experiment', ascending=True, inplace=True)
    df.reset_index(inplace=True)
    #print(df)
    predictionsDFList.append(df)
    predictionsDF = predictionsDF.add(predictionWeights[index]*df[['Experiment', 'y_trueIC', 'y_trueEmax','y_predIC', 'y_predEmax']], fill_value=0)
    #sum each prediction, weighted
    index += 1
#and divide by total number of weights
predictionsDF = predictionsDF.divide( sum(predictionWeights) )
#print(predictionsDF)

predictionsDF.to_csv(outputEnsembleFile)
print("Predictions written to file!")