import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os
from src.graphFunctions import regressionGraphs, barplot, stackedGroupedbarplot


##########################
##########################

#BARPLOT DRAW SETTINGS#####
barplotXSize = 12 #x and y sizes to render the graphs on (smaller x makes bars thinner)
barplotYSize = 14
drawBarPlotValues = False #whether to draw numeric values of each bar in barplot figures
groupedBarplotLegendLocation = 'lower center' #upper, center, lower + left, center, right (or just 'center')    - for grouped and stacked grouped barplots
tiltAngle = 45
title = '       Replicated Model Performance'
titleOffset = 30
###########################

runStackedGraphs = False #whether to generate stacked graphs
generateRegressionGraphs = False #whether to generate regression graphs

#Almanac
predictionPaths = [Path(__file__).parent / 'predictions' / 'almanac' / 'predictionsDLTheir.csv', Path(__file__).parent / 'predictions' /'almanac'/ 'predictionsEN.csv', Path(__file__).parent / 'predictions' /'almanac'/ 'predictionsLinearSVR.csv', Path(__file__).parent / 'predictions' /'almanac'/ 'predictionsRFTheir.csv', Path(__file__).parent / 'predictions' /'almanac'/ 'predictionsXGBoostJuly.csv', Path(__file__).parent / 'predictions' /'almanac'/ 'predictionsLGBMJuly.csv', Path(__file__).parent / 'predictions' / 'almanac' /  'predictionsEnsembleJulyFinal.csv', Path(__file__).parent / 'predictions' /'almanac'/'predictionsBaseline.csv']
predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'almanac'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'regular'
#groupedBars = False


##########################
##########################





##############################################
####CREATE MODEL STATISTICS FILE
##############################################
counter = 0

fullStatsDF = [] 

for j in predictionPaths:

    pred = pd.read_csv(j)

    modelName = predictionNames[counter]
    counter += 1

    rho, p = spearmanr(pred['y_true'], pred['y_pred'])
    pearson, p = pearsonr(pred['y_true'], pred['y_pred'])
    r2 = r2_score(pred['y_true'], pred['y_pred'])
    mse = mean_squared_error(pred['y_true'], pred['y_pred'])

    ar = [modelName, pearson, rho, r2, mse]
    df = pd.DataFrame(data=[ar], columns=['name', 'Pearson', 'Spearman', 'R2', 'MSE'])
    fullStatsDF.append(df)
    #df.to_csv(Path(__file__).parent / 'multiResults.csv', index=False, header=False)

fullStatsDF = pd.concat(fullStatsDF, axis=0)
print(fullStatsDF)


##############################################
####GENERATE SCATTER GRAPHS AND BAR PLOTS
##############################################
roundedOld = fullStatsDF.round(3)
fullStatsDF = fullStatsDF.set_index('name')
rounded = fullStatsDF.round(3)

#finalNames =['Model']
#for idk in stackedResultNames:
#    finalNames.append(idk)

for result in fullStatsDF.columns[0:5]: #i.e. for each of 'Pearson IC50', 'Spearman IC50', 'R2 IC50', 'MSE IC50', 'Pearson Emax', 'Spearman Emax', 'R2 Emax', 'MSE Emax'
    barplot(roundedOld, saveGraphsFolder,'almanac', resultName = result, barplotXSize=barplotXSize, barplotYSize=barplotYSize,drawBarPlotValues=drawBarPlotValues, groupedBarplotLegendLocation=groupedBarplotLegendLocation, tiltAngle=tiltAngle, title=title, titleOffset=titleOffset)

