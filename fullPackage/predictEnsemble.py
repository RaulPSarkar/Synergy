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
#Change
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DLrun1.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv']
predictionWeights = [1,1,1]
outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensemble.csv'
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