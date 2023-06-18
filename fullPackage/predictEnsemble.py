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
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun0.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DL.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun0.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun0.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun0.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun0.csv']
almanac = False
##########################
##########################


#ID is: Cell line, drug A, drug B (or just make ID var)


if(not almanac):


    for path in predictionPaths():
        pass

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