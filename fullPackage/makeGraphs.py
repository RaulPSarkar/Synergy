import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os
from src.graphFunctions import regressionGraphs, barplot, stackedbarplot



##########################
##########################

runStackedGraphs = False #whether to generate stacked graphs

#Regular
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun3.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun0.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99Regular.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensemble.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'regular'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'regular'
#groupedBars = False


#Shuffled
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun6.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun33.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun33.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun33.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun33.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun33.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensembleShuffled.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'shuffled'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'shuffled'
#groupedBars = False


#Misc Results
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99Regular.csv', Path(__file__).parent / 'predictions' / 'temp' / 'dlNew1JustCoeffs.csv',  Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99regularplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun100regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun100regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'temp' /'rf0CoeffsNoSingle.csv']
#predictionNames = ['DL', 'LGBM','DL Coeffs', 'LGBM Coeffs', 'LGBM Coeffs+Single', 'RF', 'RF Coeffs+Single', 'RF Coeffs']
#mainModel = ['DL', 'LGBM', 'DL', 'LGBM', 'LGBM', 'RF', 'RF', 'RF']
#type = ['Drugs', 'Drugs', 'Coeffs', 'Coeffs', 'Coeffs+Single', 'Drugs', 'Coeffs+Single', 'Coeffs']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'misc'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'misc'
#groupedBars = True



#Omics Type
predictionPaths = [Path(__file__).parent / 'predictions' /'temp'/ 'lgbm0GE.csv',Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun100regularplusSingleplusCoeffs.csv',  Path(__file__).parent / 'predictions' /'temp' /'lgbm0Proteomics.csv']
predictionNames = ['LGBM Coeffs+Single GE', 'LGBM Coeffs+Single CRISPR', 'LGBM Coeffs+Single Proteomics']
mainModel = ['LGBM', 'LGBM', 'LGBM']
type = ['Transcriptomics', 'CRISPR-KO', 'Proteomics']
saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'omicstype'
modelStatsFolder =  Path(__file__).parent / 'results' / 'omicstype'
groupedBars = True


#Drug CV
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1drug.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun3drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun1drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensembleDrug.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'drugCV'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'drugCV'
#groupedBars = False


#Cell CV
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1cell.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun3cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun1cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensembleCell.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'cellCV'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'cellCV'
#groupedBars = False


stackedResults = [Path(__file__).parent / 'results' / 'shuffled' / 'results.csv', Path(__file__).parent / 'results' / 'regular'/ 'results.csv']
stackedResultNames = ['Shuffled', 'Results']
stackedGraphsFolder = Path(__file__).parent / 'graphs' / 'stackedBarPlots' / 'shuffled'

#stackedResults = [Path(__file__).parent / 'results' / 'cellCV' / 'results.csv', Path(__file__).parent / 'results' / 'drugCV' / 'results.csv', Path(__file__).parent / 'results' / 'regular'/ 'results.csv']
#stackedResultNames = ['Cell CV', 'Drug CV', 'Regular CV']
#stackedGraphsFolder = Path(__file__).parent / 'graphs' / 'stackedBarPlots' / 'CV'


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

if not os.path.exists(stackedGraphsFolder):
    os.mkdir(stackedGraphsFolder)


fullStatsDF.to_csv(modelStatsFolder / 'results.csv', index=False)
print(fullStatsDF)


##############################################
####GENERATE SCATTER GRAPHS AND BAR PLOTS
##############################################
roundedOld = fullStatsDF.round(3)
fullStatsDF = fullStatsDF.set_index('name')
rounded = fullStatsDF.round(3)

if(groupedBars):
    roundedOld['mainModel'] = mainModel
    roundedOld['type'] = type


finalNames =['Model']
for idk in stackedResultNames:
    finalNames.append(idk)


cumulativeResults = None

if(runStackedGraphs):
    for result in fullStatsDF.columns:
        fullResults = []
        fullResults.append(pd.DataFrame(predictionNames, columns=['Model']) )
        
        count = 0
        for j in stackedResults:
            res = pd.read_csv(j)
            if(count==0):
                fullResults.append(res[result])
            else:
                fullResults.append(res[result] - previousResult)
            previousResult = res[result]
            count += 1

        shuffledFingerprintTable = pd.concat(fullResults, axis=1)
        shuffledFingerprintTable.columns = finalNames
        print(shuffledFingerprintTable)
        stackedbarplot(shuffledFingerprintTable, stackedGraphsFolder ,result)







counter = 0
for j in predictionPaths:

    df = pd.read_csv(j)
    modelName = predictionNames[counter]
    regressionGraphs(df, modelName, rounded, saveGraphsFolder)

    counter += 1



for result in fullStatsDF.columns:
    barplot(roundedOld, saveGraphsFolder,'gdsc', resultName = result, groupedBars = groupedBars)

