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
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun1.csv']#, Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun0.csv', ]
outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensemble.csv'
almanac = False
##########################
##########################


#ID is Experiment column


if(not almanac):


    predictionsDFList = []
    predictionsDF = pd.DataFrame()

    for path in predictionPaths:
        df = pd.read_csv(path)
        df.sort_values(by='Experiment', ascending=True, inplace=True)
        df.reset_index(inplace=True)
        print(df)
        predictionsDFList.append(df)
        predictionsDF = predictionsDF.add(df[['Experiment', 'y_trueIC', 'y_trueEmax','y_predIC', 'y_predEmax']], fill_value=0)
        #sum each prediction

    #and divide by total number of predictions to obtain the mean
    predictionsDF = predictionsDF.divide( len(predictionPaths) )
    print(predictionsDF)

    predictionsDF.to_csv(outputEnsembleFile)