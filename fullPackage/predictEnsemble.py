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

    print(predictionsDF)

    rf = pd.read_csv('predictions/MultiNoConc/predictionsExpressionRF.csv')
    svr = pd.read_csv('predictions/MultiNoConc/predictionsMultiExpressionSVR.csv')
    en = pd.read_csv('predictions/MultiNoConc/predictionsMultiExpressionEN.csv')
    dl = pd.read_csv('predictions/MultiNoConc/predictionsExpressionDL.csv')
    #baseline = pd.read_csv('predictions/MultiNoConc/predictionsBaseline.csv')
    xgb = pd.read_csv('predictions/MultiNoConc/predictionsMultiExpressionXGB.csv')
    lgbm = pd.read_csv('predictions/MultiNoConc/predictionsMultiExpressionLGBM.csv')


    df = rf


    df['y_predIC'] = (rf['y_predIC']+ svr['y_predIC']+ en['y_predIC']+dl['y_predIC']  +xgb['y_predIC']  + lgbm['y_predIC']) / 6
    df['y_predEmax'] = (rf['y_predEmax']+ svr['y_predEmax']+ en['y_predEmax']+dl['y_predEmax']  +xgb['y_predEmax']  + lgbm['y_predEmax']) / 6
    df.to_csv('predictions/MultiNoConc/predictionsEnsembleNoBase.csv')

else:
    rf = pd.read_csv('predictions/Almanac/predictionsRFTheir.csv')
    svr = pd.read_csv('predictions/Almanac/predictionsLinearSVR.csv')
    en = pd.read_csv('predictions/Almanac/predictionsEN.csv')
    dl = pd.read_csv('predictions/Almanac/predictionsDLTheir.csv')
    xgb = pd.read_csv('predictions/Almanac/predictionsXGB.csv')
    lgbm = pd.read_csv('predictions/Almanac/predictionsLGBM.csv')
    #baseline = pd.read_csv('predictions/almanac/predictionsBaseline.csv')

    df = pd.DataFrame()
    df['y_true'] = rf['y_true']
    df['y_pred'] = (rf['y_pred']+ svr['y_pred']+ en['y_pred']+dl['y_pred'] +xgb['y_pred']  + lgbm['y_pred']) / 6
    

    rho, p = spearmanr(rf['y_true'], df['y_pred'])
    pearson, p = pearsonr(rf['y_true'], df['y_pred'])
    r2 = r2_score(rf['y_true'], df['y_pred'])
    mse = mean_squared_error(rf['y_true'], df['y_pred'])

    print(rho, pearson, r2, mse)
    df.to_csv('predictions/Almanac/predictionsEnsemble.csv')