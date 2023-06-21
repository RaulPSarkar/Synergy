import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os
from src.graphFunctions import regressionGraphs, barplot



##########################
##########################
#Change
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun0.csv', Path(__file__).parent / 'predictions' /'predictions.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun0.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun0.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun0.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun0.csv']
predictionNames = ['Baseline', 'EN', 'DL', 'LGBM', 'RF', 'XGBoost', 'SVR']
saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'regular'
modelStatsFolder =  Path(__file__).parent / 'results'
##########################
##########################





##############################################
####CREATE MODEL STATISTICS FILE
##############################################
counter = 0

fullStatsDF = [] 

for j in predictionPaths:

    pred = pd.read_csv(j)
    pred = pred.dropna(subset=['y_trueIC','y_predIC','y_trueEmax','y_predEmax'])
    #NAs dropped (they shouldn't exist) just in case the baseline has never seen the test drug pair

    modelName = predictionNames[counter]
    counter += 1



    rhoEmax, p = spearmanr(pred['y_trueEmax'], pred['y_predEmax'])
    rhoIC50, p = spearmanr(pred['y_trueIC'], pred['y_predIC'])
    pearsonIC50, p = pearsonr(pred['y_trueIC'], pred['y_predIC'])
    pearsonEmax, p = pearsonr(pred['y_trueEmax'], pred['y_predEmax'])
    r2IC50 = r2_score(pred['y_trueIC'], pred['y_predIC'])
    r2Emax = r2_score(pred['y_trueEmax'], pred['y_predEmax'])
    mseIC50 = mean_squared_error(pred['y_trueIC'], pred['y_predIC'])
    mseEmax = mean_squared_error(pred['y_trueEmax'], pred['y_predEmax'])
    rho, p = spearmanr(pred[['y_trueIC', 'y_trueEmax']], pred[['y_predIC', 'y_predEmax']], axis=None)
    r2 = r2_score(pred[['y_trueIC', 'y_trueEmax']], pred[['y_predIC', 'y_predEmax']])

    ar = [modelName, pearsonIC50, rhoIC50, r2IC50, mseIC50, pearsonEmax, rhoEmax, r2Emax, mseEmax]
    df = pd.DataFrame(data=[ar], columns=['name', 'Pearson IC50', 'Spearman IC50', 'R2 IC50', 'MSE IC50', 'Pearson Emax',  'Spearman Emax', 'R2 Emax', 'MSE Emax'])
    fullStatsDF.append(df)
    #df.to_csv(Path(__file__).parent / 'multiResults.csv', index=False, header=False)

fullStatsDF = pd.concat(fullStatsDF, axis=0)

if not os.path.exists(modelStatsFolder):
    os.mkdir(modelStatsFolder)

fullStatsDF.to_csv(modelStatsFolder / 'results.csv', index=False)
print(fullStatsDF)


##############################################
####GENERATE SCATTER GRAPHS AND BAR PLOTS
##############################################
roundedOld = fullStatsDF.round(4)
fullStatsDF = fullStatsDF.set_index('name')
rounded = fullStatsDF.round(4)

counter = 0

for j in predictionPaths:

    df = pd.read_csv(j)
    modelName = predictionNames[counter]
    regressionGraphs(df, modelName, rounded, saveGraphsFolder)

    counter += 1

barplot(roundedOld, saveGraphsFolder,'gdsc')

print("Graphs written to disk!")
