import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os
from src.graphFunctions import regressionGraphs, barplot, stackedGroupedbarplot


##########################
##########################

#BARPLOT DRAW SETTINGS#####
barplotXSize = 10 #x and y sizes to render the graphs on (smaller x makes bars thinner)
barplotYSize = 16
drawBarPlotValues = False #whether to draw numeric values of each bar in barplot figures
groupedBarplotLegendLocation = 'lower center' #upper, center, lower + left, center, right (or just 'center')    - for grouped and stacked grouped barplots
tiltAngle = 45
title = ''
###########################

runStackedGraphs = False #whether to generate stacked graphs
generateRegressionGraphs = False #whether to generate regression graphs

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


#Shuffled Grouped
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun3.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun0.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99Regular.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensemble.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv', Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun6.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun33.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun33.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun33.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun33.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun33.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensembleShuffled.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline', 'DL Shuffled', 'EN Shuffled', 'SVR Shuffled' ,'RF Shuffled', 'XGBoost Shuffled', 'LGBM Shuffled', 'Ensemble Shuffled', 'Baseline Shuffled']
#mainModel = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline', 'DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#type = ['Morgan Fingerprints', 'Morgan Fingerprints', 'Morgan Fingerprints', 'Morgan Fingerprints', 'Morgan Fingerprints', 'Morgan Fingerprints', 'Morgan Fingerprints', 'Morgan Fingerprints', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits', 'Fingerprints with Shuffled Bits']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'shuffledGrouped'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'shuffledGrouped'
#groupedBars = True



#Misc Results
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1.csv', Path(__file__).parent / 'predictions' / 'final' / 'dlCoeffs' / 'dlCoeffsrun111regulargeplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111regularplusSingleplusCoeffs.csv',  Path(__file__).parent / 'predictions' / 'final' / 'dlCoeffs' / 'dlCoeffsrun111regularplusSingleplusCoeffsplusCType.csv',   Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99Regular.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99regularplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun100regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun100regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun114regulargeplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'rf' /'rfrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' / 'final' / 'xgboost' / 'xgboostrun99.csv', Path(__file__).parent / 'predictions' / 'final' / 'xgboost' / 'xgboostrun115regulargeplusCoeffs.csv', Path(__file__).parent / 'predictions' / 'final' / 'xgboost' / 'xgboostrun115regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' / 'final' / 'xgboost' / 'xgboostrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' / 'ensemble.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' / 'ensembleSinglePlusCoeffsPlusType.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' / 'ensembleSinglePlusCoeffs.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' / 'ensembleCoeffs.csv']
#predictionNames = ['DL', 'DL Coeffs', 'DL Coeffs+Single', 'DL CancerType',   'LGBM', 'LGBM Coeffs', 'LGBM Coeffs+Single', 'RF', 'RF Coeffs+Single', 'RF Coeffs', 'LGBM CancerType', 'RF CancerType', 'XGBoost', 'XGBoost Coeffs', 'XGBoost Coeffs+Single', 'XGBoost CancerType', 'Ensemble', 'Ensemble CancerType', 'Ensemble Coeffs+Single', 'Ensemble Coeffs']
#mainModel = ['DL', 'DL', 'DL', 'DL', 'LGBM', 'LGBM', 'LGBM', 'RF', 'RF', 'RF', 'LGBM', 'RF',  'XGBoost', 'XGBoost', 'XGBoost', 'XGBoost', 'Ensemble', 'Ensemble', 'Ensemble', 'Ensemble']
#type = ['Fingerprints', 'Coeffs', 'Coeffs+Single Agent', 'Coeffs+Single Agent + Cancer Type',   'Fingerprints', 'Coeffs', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Coeffs', 'Coeffs+Single Agent + Cancer Type', 'Coeffs+Single Agent + Cancer Type',  'Fingerprints', 'Coeffs', 'Coeffs+Single Agent', 'Coeffs+Single Agent + Cancer Type', 'Fingerprints', 'Coeffs+Single Agent + Cancer Type', 'Coeffs+Single Agent', 'Coeffs']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'misc'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'misc'
#groupedBars = True



#Omics Type
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun117regularge.csv',Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun117regularcrispr.csv',  Path(__file__).parent / 'predictions' /'final'/'lgbm' /'lgbmrun117regularproteomics.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun115regularge.csv',Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun115regularcrispr.csv',  Path(__file__).parent / 'predictions' /'final'/'rf' /'rfrun115regularproteomics.csv']
#predictionNames = ['LGBM Coeffs+Single GE', 'LGBM Coeffs+Single CRISPR', 'LGBM Coeffs+Single Proteomics', 'RF Coeffs+Single GE', 'RF Coeffs+Single CRISPR', 'RF Coeffs+Single Proteomics']
#mainModel = ['LGBM', 'LGBM', 'LGBM', 'RF', 'RF', 'RF']
#type = ['Transcriptomics', 'CRISPR-KO', 'Proteomics', 'Transcriptomics', 'CRISPR-KO', 'Proteomics']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'omicstype'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'omicstype'
#groupedBars = True

#Drug CV
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1drug.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun3drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun1drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensembleDrug.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'drugCV'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'drugCV'
#groupedBars = False

#Regular Comparison
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99Regular.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun115regularplusSingleplusCoeffsplusCType.csv',Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'rf' /'rfrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost' /'xgboostrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'ensemble' /'ensemble.csv', Path(__file__).parent / 'predictions' /'final'/'ensemble' /'ensembleSinglePlusCoeffsPlusType.csv']
#predictionNames = ['LGBM Drugs', 'LGBM Coeffs+Single', 'RF Drugs', 'RF Coeffs+Single', 'DL Drugs', 'DL Coeffs+Single', 'XGBoost Drugs', 'XGBoost Coeffs+Single', 'Ensemble Drugs', 'Ensemble Coeffs+Single']
#mainModel = [ 'LGBM', 'LGBM', 'RF', 'RF', 'DL', 'DL', 'XGBoost', 'XGBoost', 'Ensemble', 'Ensemble']
#type = ['Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'regularComparison'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'regularComparison'
#groupedBars = True

#Cell CV Comparison
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99cellplusDrugs.csv',Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun116cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun115cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1cell.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost' /'xgboostrun115cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'ensemble' /'ensembleCell.csv', Path(__file__).parent / 'predictions' /'final'/'ensemble' /'ensembleCellSinglePlusCoeffsPlusType.csv']
#predictionNames = ['LGBM Drugs', 'LGBM Coeffs+Single', 'RF Drugs', 'RF Coeffs+Single', 'DL Drugs', 'DL Coeffs+Single', 'XGBoost Drugs', 'XGBoost Coeffs+Single', 'Ensemble Drugs', 'Ensemble Coeffs+Single']
#mainModel = [ 'LGBM', 'LGBM', 'RF', 'RF', 'DL', 'DL', 'XGBoost', 'XGBoost', 'Ensemble', 'Ensemble']
#type = ['Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'cellCVcomparison'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'cellCVcomparison'
#groupedBars = True


#Drug CV Comparison
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99drugplusDrugs.csv',Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun117drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun115drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1drug.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost' /'xgboostrun115drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'ensemble' /'ensembleDrug.csv', Path(__file__).parent / 'predictions' /'final'/'ensemble' /'ensembleDrugSinglePlusCoeffsPlusType.csv']
#predictionNames = ['LGBM Drugs', 'LGBM Coeffs+Single', 'RF Drugs', 'RF Coeffs+Single', 'DL Drugs', 'DL Coeffs+Single', 'XGBoost Drugs', 'XGBoost Coeffs+Single', 'Ensemble Drugs', 'Ensemble Coeffs+Single']
#mainModel = [ 'LGBM', 'LGBM', 'RF', 'RF', 'DL', 'DL', 'XGBoost', 'XGBoost', 'Ensemble', 'Ensemble']
#type = ['Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent', 'Fingerprints', 'Coeffs+Single Agent']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'drugCVcomparison'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'drugCVcomparison'
#groupedBars = True


#Cell CV
#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'DL' / 'dlrun1cell.csv', Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun3cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'svr'/ 'svrrun1cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/ 'rf'/ 'rfrun11cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'ensemble' /  'ensembleCell.csv', Path(__file__).parent / 'predictions' /'final'/'baseline'/ 'baselinerun0.csv']
#predictionNames = ['DL', 'EN', 'SVR' ,'RF', 'XGBoost', 'LGBM', 'Ensemble', 'Baseline']
#saveGraphsFolder =  Path(__file__).parent / 'graphs' / 'cellCV'
#modelStatsFolder =  Path(__file__).parent / 'results' / 'cellCV'
#groupedStackedBars = False


#stackedResults = [Path(__file__).parent / 'results' / 'shuffled' / 'results.csv', Path(__file__).parent / 'results' / 'regular'/ 'results.csv']
#stackedResultNames = ['Shuffled', 'Results']
#stackedGraphsFolder = Path(__file__).parent / 'graphs' / 'stackedBarPlots' / 'shuffled'

#stackedResults = [Path(__file__).parent / 'results' / 'cellCV' / 'results.csv', Path(__file__).parent / 'results' / 'drugCV' / 'results.csv', Path(__file__).parent / 'results' / 'regular'/ 'results.csv']
#stackedResultNames = ['Cell CV', 'Drug CV', 'Regular CV']
#stackedGraphsFolder = Path(__file__).parent / 'graphs' / 'stackedBarPlots' / 'CV'


stackedResults = [Path(__file__).parent / 'results' / 'cellCVcomparison' / 'results.csv', Path(__file__).parent / 'results' / 'drugCVcomparison' / 'results.csv', Path(__file__).parent / 'results' / 'regularComparison'/ 'results.csv']
stackedResultNames = ['Cell CV', 'Drug CV', 'Regular CV']
stackedGraphsFolder = Path(__file__).parent / 'graphs' / 'stackedGroupedBarPlots' / 'CV'


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


if(groupedBars):
    fullStatsDF['mainModel'] = mainModel
    fullStatsDF['type'] = type


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

print(roundedOld)
cumulativeResults = None

if(runStackedGraphs):

    stackedDifferences = pd.DataFrame()

    for result in fullStatsDF.columns[0:8]:
        print(result)

    #    if(mainModel,type):
    #        skip
        fullResults = []
        fullResults.append(pd.DataFrame(predictionNames, columns=['Model']) )

        

        count = 0
        for j in stackedResults:
            res = pd.read_csv(j)

            aaaa = res['mainModel'] 
            bbbb = res['type'] 

            if(count==0):
                fullResults.append(res[result])
            else:
                fullResults.append(res[result] - previousResult)
            previousResult = res[result]
            count += 1

        shuffledFingerprintTable = pd.concat(fullResults, axis=1)
        shuffledFingerprintTable.columns = finalNames
        #stackedbarplot(shuffledFingerprintTable, stackedGraphsFolder ,result)
############ OLD PORTION ended here ##########

        shuffledFingerprintTable['mainModel'] = aaaa
        shuffledFingerprintTable['type'] = bbbb

        #print(shuffledFingerprintTable)

        shuffledFingerprintTable = shuffledFingerprintTable.drop(['Model'], axis=1)
        #print(shuffledFingerprintTable)

        shuffledFingerprintTable.set_index(['mainModel', 'type'], inplace=True)
        shuffledFingerprintTable = shuffledFingerprintTable.reorder_levels(['mainModel', 'type']).sort_index()

        shuffledFingerprintTable = shuffledFingerprintTable.unstack(level=-1) # unstack the 'Context' column
        #print(shuffledFingerprintTable)

        CellDifferences = shuffledFingerprintTable["Cell CV"]
        CellDifferences = CellDifferences["Coeffs+Single Agent"] - CellDifferences["Fingerprints"]
        DrugDifferences = shuffledFingerprintTable["Drug CV"] + shuffledFingerprintTable["Cell CV"]
        DrugDifferences = DrugDifferences["Coeffs+Single Agent"] - DrugDifferences["Fingerprints"]
        CVDifferences = shuffledFingerprintTable["Regular CV"] + shuffledFingerprintTable["Drug CV"] + shuffledFingerprintTable["Cell CV"]
        CVDifferences = CVDifferences["Coeffs+Single Agent"] - CVDifferences["Fingerprints"]

        stackedDifferences['Cell'] = CellDifferences
        stackedDifferences['Drug'] = DrugDifferences
        stackedDifferences['CV'] = CVDifferences


        if( ('Pearson' in result) or ('Spearman' in result) or ('R2' in result)):
            fileName = result + 'Differences.csv'
            stackedDifferences.to_csv(modelStatsFolder / fileName, index=False)

        
        #print(shuffledFingerprintTable["Drug CV"] + shuffledFingerprintTable["Cell CV"])
        #print(shuffledFingerprintTable["Regular CV"] + shuffledFingerprintTable["Drug CV"] + shuffledFingerprintTable["Cell CV"])

        #shuffledFingerprintTable['mainModel'] = fullStatsDF['mainModel']
        #shuffledFingerprintTable['type'] = fullStatsDF['mainModel']
        #print(shuffledFingerprintTable)

        stackedGroupedbarplot(shuffledFingerprintTable, stackedGraphsFolder ,result, barplotXSize=barplotXSize, barplotYSize=barplotYSize, groupedBarplotLegendLocation=groupedBarplotLegendLocation)






if(not runStackedGraphs):
    if(generateRegressionGraphs):
        counter = 0
        for j in predictionPaths:

            df = pd.read_csv(j)
            modelName = predictionNames[counter]
            regressionGraphs(df, modelName, rounded, saveGraphsFolder)

            counter += 1


    for result in fullStatsDF.columns[0:8]: #i.e. for each of 'Pearson IC50', 'Spearman IC50', 'R2 IC50', 'MSE IC50', 'Pearson Emax', 'Spearman Emax', 'R2 Emax', 'MSE Emax'
        barplot(roundedOld, saveGraphsFolder,'gdsc', resultName = result, groupedBars = groupedBars, barplotXSize=barplotXSize, barplotYSize=barplotYSize,drawBarPlotValues=drawBarPlotValues, groupedBarplotLegendLocation=groupedBarplotLegendLocation, tiltAngle=tiltAngle, title=title)

