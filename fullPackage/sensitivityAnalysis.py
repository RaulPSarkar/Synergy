import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun70regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun71regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun72regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun73regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun74regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun75regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun76regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun77regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun78regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun79regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun710regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun711regularplusDrugs.csv']
sensitivitySizeFractions = [0.01, 0.03, 0.06, 0.125, 0.17, 0.25, 0.375, 0.5, 0.625, 0.75, 0.85, 1] #sizes used for training (used to plot the model)
resultsFolder =  Path(__file__).parent / 'results' / 'sens'
graphsFolder =  Path(__file__).parent / 'graphs' / 'sens'






#this is copied from makeGraphs, turn it into a function later
##############################################
####CREATE MODEL STATISTICS FILE
##############################################
counter = 0

fullStatsDF = [] 

for j in predictionPaths:

    pred = pd.read_csv(j)
    pred = pred.dropna(subset=['y_trueIC','y_predIC','y_trueEmax','y_predEmax'])
    #NAs dropped (they shouldn't exist) just in case the baseline has never seen the test drug pair

    modelName = sensitivitySizeFractions[counter]
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

if not os.path.exists(resultsFolder):
    os.mkdir(resultsFolder)

if not os.path.exists(graphsFolder):
    os.mkdir(graphsFolder)



fullStatsDF.to_csv(resultsFolder / 'results.csv', index=False)
print(fullStatsDF)



for metricName in ['Pearson IC50', 'Spearman IC50', 'R2 IC50', 'Spearman Emax', 'R2 Emax', 'MSE Emax']:
    

    
    sns.set_style('whitegrid')
    plot=sns.lineplot(data=fullStatsDF, x="name", y="Pearson IC50")

    plot.set(xscale='log')
    plot.set(xticks=sensitivitySizeFractions)
    plot.set(xticklabels=sensitivitySizeFractions)
    plot.set(yticks=np.linspace(0, 1, 11))

    figure = plt.gcf()
    figure.set_size_inches(32, 18)
    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)
    plt.xlabel("Fraction of Dataset used to Train/Test Model", size=36)
    plt.title('Model Performance per Dataset Size', size=44)
    plt.ylabel(metricName, size=36)

    fileName = 'sizeSensAnalysis' + metricName + '.png'
    plt.savefig(graphsFolder / fileName)
    plt.close()

