import pandas as pd
from pathlib import Path
import os
from scipy.stats import f_oneway
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


#One-Way ANOVA (to demonstrate that omics branch makes no difference)

drawBoxplotGraphs = True #to show performance of each fold
plotXSize = 10
plotYSize = 16
graphFolder = Path(__file__).parent / '../graphs' / 'statisticalSignificance' / 'RF'

useFoldFiles = True #if set to true, it will not divide up final file into folds

#FOR OMICS PVALUE
basePaths1 = [Path(__file__).parent / '../predictions' /'temp'/ 'lgbmrun117regularcrispr', Path(__file__).parent / '../predictions' /'temp'/ 'lgbmrun117regularge', Path(__file__).parent / '../predictions' /'temp'/ 'lgbmrun117regularproteomics']
basePaths2 = [Path(__file__).parent / '../predictions' /'temp'/ 'rfrun115regularcrispr', Path(__file__).parent / '../predictions' /'temp'/ 'rfrun115regularge', Path(__file__).parent / '../predictions' /'temp'/ 'rfrun115regularproteomics']
groupOfPaths = [basePaths1, basePaths2]
groupOfModels = ['LGBM', 'RF']

resultsFile =  Path(__file__).parent / '../results' / 'statisticalSignificance' / 'omicsSignificance.csv'


#FOR MODEL COMPARISON PVALUE
#lgbmDrugs = [Path(__file__).parent / '../predictions' /'final'/'lgbm'/ 'lgbmrun99drugplusDrugs.csv',Path(__file__).parent / '../predictions' /'final'/'lgbm'/ 'lgbmrun117drugplusSingleplusCoeffsplusCType.csv']
#rfDrugs = [Path(__file__).parent / '../predictions' /'final'/ 'rf'/ 'rfrun11drugplusDrugs.csv', Path(__file__).parent / '../predictions' /'final'/ 'rf'/ 'rfrun115drugplusSingleplusCoeffsplusCType.csv']
#dlDrugs = [Path(__file__).parent / '../predictions' / 'final' / 'DL' / 'dlrun1drug.csv', Path(__file__).parent / '../predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111drugplusSingleplusCoeffsplusCType.csv']
#xgboostDrugs = [Path(__file__).parent / '../predictions' /'final'/'xgboost'/ 'xgboostrun99drugplusDrugs.csv', Path(__file__).parent / '../predictions' /'final'/'xgboost' /'xgboostrun115drugplusSingleplusCoeffsplusCType.csv']
#ensembleDrugs = [Path(__file__).parent / '../predictions' /'final'/'ensemble' /'ensembleDrug.csv', Path(__file__).parent / '../predictions' /'final'/'ensemble' /'ensembleDrugSinglePlusCoeffsPlusType.csv']

#lgbmCell = [Path(__file__).parent / '../predictions' /'final'/'lgbm'/ 'lgbmrun99cellplusDrugs.csv',Path(__file__).parent / '../predictions' /'final'/'lgbm'/ 'lgbmrun116cellplusSingleplusCoeffsplusCType.csv']
#rfCell = [Path(__file__).parent / '../predictions' /'final'/ 'rf'/ 'rfrun11cellplusDrugs.csv', Path(__file__).parent / '../predictions' /'final'/ 'rf'/ 'rfrun115cellplusSingleplusCoeffsplusCType.csv']
#dlCell = [Path(__file__).parent / '../predictions' / 'final' / 'DL' / 'dlrun1cell.csv', Path(__file__).parent / '../predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111cellplusSingleplusCoeffsplusCType.csv']
#xgboostCell = [Path(__file__).parent / '../predictions' /'final'/'xgboost'/ 'xgboostrun99cellplusDrugs.csv', Path(__file__).parent / '../predictions' /'final'/'xgboost' /'xgboostrun115cellplusSingleplusCoeffsplusCType.csv']
#ensembleCell = [Path(__file__).parent / '../predictions' /'final'/'ensemble' /'ensembleCell.csv', Path(__file__).parent / '../predictions' /'final'/'ensemble' /'ensembleCellSinglePlusCoeffsPlusType.csv']

#lgbmRegular = [Path(__file__).parent / '../predictions' /'final'/'lgbm'/ 'lgbmrun99Regular.csv', Path(__file__).parent / '../predictions' /'final'/'lgbm'/ 'lgbmrun115regularplusSingleplusCoeffsplusCType.csv']
#rfRegular = [Path(__file__).parent / '../predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / '../predictions' /'final'/'rf' /'rfrun115regularplusSingleplusCoeffsplusCType.csv']
#dlRegular = [Path(__file__).parent / '../predictions' / 'final' / 'DL' / 'dlrun1.csv', Path(__file__).parent / '../predictions' /'final'/'dlCoeffs' /'dlCoeffsrun111regularplusSingleplusCoeffsplusCType.csv']
#xgboostRegular= [Path(__file__).parent / '../predictions' /'final'/'xgboost'/ 'xgboostrun99.csv', Path(__file__).parent / '../predictions' /'final'/'xgboost' /'xgboostrun115regularplusSingleplusCoeffsplusCType.csv']
#ensembleRegular = [Path(__file__).parent / '../predictions' /'final'/'ensemble' /'ensemble.csv', Path(__file__).parent / '../predictions' /'final'/'ensemble' /'ensembleSinglePlusCoeffsPlusType.csv']

#groupOfPaths = [lgbmDrugs, rfDrugs, dlDrugs, xgboostDrugs, ensembleDrugs, lgbmCell, rfCell, dlCell, xgboostCell, ensembleCell,  lgbmRegular, rfRegular, dlRegular, xgboostRegular, ensembleRegular]
#groupOfModels = ['LGBM Drugs', 'RF Drugs', 'DL Drugs', 'XGBoost Drugs', 'Ensemble Drugs', 'LGBM Cell', 'RF Cell', 'DL Cell', 'XGBoost Cell', 'Ensemble Cell', 'LGBM Regular', 'RF Regular', 'DL Regular', 'XGBoost Regular', 'Ensemble Regular']
#resultsFile =  Path(__file__).parent / '../results' / 'statisticalSignificance' / 'modelSignificance.csv'

numberOfFolds = 5 #number of folds used during train/test












def foldSeparation(df, numberOfFolds):
    return np.array_split(df, numberOfFolds)
    #folds are saved to DF as they are obtained, so we can just separate them (since I didn't save fold by fold for the most part)




fullStatsDF = [] 



ind = 0
for basePaths in groupOfPaths:
    modelName = groupOfModels[ind]

    pearsonIC50Matrix = []
    pearsonEmaxMatrix = []

    mseIC50Matrix = []
    mseEmaxMatrix = []

    rhoIC50Matrix = []
    rhoEmaxMatrix = []

    r2IC50Matrix = []
    r2EmaxMatrix = []



    for basePath in basePaths:
        pearsonIC50Array = []
        pearsonEmaxArray = []
        mseIC50Array = []
        mseEmaxArray = []
        r2IC50Array = []
        r2EmaxArray = []
        rhoIC50Array = []
        rhoEmaxArray = []



        if(not useFoldFiles):

            predList = pd.read_csv(basePath)
            predList = foldSeparation(predList, numberOfFolds)
            
            
        for fold in range(numberOfFolds):
            
            if(useFoldFiles):
                path = str(basePath)
                fullPath = path + str(fold) + '.csv'
                pred = pd.read_csv(fullPath)
                pred = pred.dropna(subset=['y_trueIC','y_predIC','y_trueEmax','y_predEmax'])
            else:
                pred = predList[fold]
                pred = pred.dropna(subset=['y_trueIC','y_predIC','y_trueEmax','y_predEmax'])



        #modelName = predictionNames[counter]
        #counter += 1
            pearsonEmax, p = pearsonr(pred['y_trueEmax'], pred['y_predEmax'])
            pearsonEmaxArray.append(pearsonEmax)
            pearsonIC50, p = pearsonr(pred['y_trueIC'], pred['y_predIC'])
            pearsonIC50Array.append(pearsonIC50)
            rhoEmax, p = spearmanr(pred['y_trueEmax'], pred['y_predEmax'])
            rhoEmaxArray.append(rhoEmax)
            rhoIC50, p = spearmanr(pred['y_trueIC'], pred['y_predIC'])
            rhoIC50Array.append(rhoIC50)
            r2IC50 = r2_score(pred['y_trueIC'], pred['y_predIC'])
            r2IC50Array.append(r2IC50)
            r2Emax = r2_score(pred['y_trueEmax'], pred['y_predEmax'])
            r2EmaxArray.append(r2Emax)
            mseIC50 = mean_squared_error(pred['y_trueIC'], pred['y_predIC'])
            mseIC50Array.append(mseIC50)
            mseEmax = mean_squared_error(pred['y_trueEmax'], pred['y_predEmax'])
            mseEmaxArray.append(mseEmax)

        pearsonIC50Matrix.append(pearsonIC50Array)
        pearsonEmaxMatrix.append(pearsonEmaxArray)

        rhoIC50Matrix.append(rhoIC50Array)
        rhoEmaxMatrix.append(rhoEmaxArray)

        r2IC50Matrix.append(r2IC50Array)
        r2EmaxMatrix.append(r2EmaxArray)

        mseIC50Matrix.append(mseIC50Array)
        mseEmaxMatrix.append(mseEmaxArray)


    if(useFoldFiles):
        pearsonEmaxPValue = f_oneway(pearsonEmaxMatrix[0], pearsonEmaxMatrix[1], pearsonEmaxMatrix[2]).pvalue
        pearsonIC50PValue = f_oneway(pearsonIC50Matrix[0], pearsonIC50Matrix[1], pearsonIC50Matrix[2]).pvalue

        rhoEmaxPValue = f_oneway(rhoEmaxMatrix[0], rhoEmaxMatrix[1], rhoEmaxMatrix[2]).pvalue
        rhoIC50PValue = f_oneway(rhoIC50Matrix[0], rhoIC50Matrix[1], rhoIC50Matrix[2]).pvalue

        r2EmaxPValue = f_oneway(r2EmaxMatrix[0], r2EmaxMatrix[1], r2EmaxMatrix[2]).pvalue
        r2IC50PValue = f_oneway(r2IC50Matrix[0], r2IC50Matrix[1], r2IC50Matrix[2]).pvalue

        mseEmaxPValue = f_oneway(mseEmaxMatrix[0], mseEmaxMatrix[1], mseEmaxMatrix[2]).pvalue
        mseIC50PValue = f_oneway(mseIC50Matrix[0], mseIC50Matrix[1], mseIC50Matrix[2]).pvalue


    else:
        pearsonEmaxPValue = f_oneway(pearsonEmaxMatrix[0], pearsonEmaxMatrix[1]).pvalue
        pearsonIC50PValue = f_oneway(pearsonIC50Matrix[0], pearsonIC50Matrix[1]).pvalue

        rhoEmaxPValue = f_oneway(rhoEmaxMatrix[0], rhoEmaxMatrix[1]).pvalue
        rhoIC50PValue = f_oneway(rhoIC50Matrix[0], rhoIC50Matrix[1]).pvalue

        r2EmaxPValue = f_oneway(r2EmaxMatrix[0], r2EmaxMatrix[1]).pvalue
        r2IC50PValue = f_oneway(r2IC50Matrix[0], r2IC50Matrix[1]).pvalue

        mseEmaxPValue = f_oneway(mseEmaxMatrix[0], mseEmaxMatrix[1]).pvalue
        mseIC50PValue = f_oneway(mseIC50Matrix[0], mseIC50Matrix[1]).pvalue



    pvalues = [modelName, pearsonIC50PValue, rhoIC50PValue, r2IC50PValue, mseIC50PValue, pearsonEmaxPValue, rhoEmaxPValue, r2EmaxPValue, mseEmaxPValue]
    df = pd.DataFrame(data=[pvalues], columns=['name', 'Pearson IC50', 'Spearman IC50', 'R2 IC50', 'MSE IC50', 'Pearson Emax',  'Spearman Emax', 'R2 Emax', 'MSE Emax'])
    fullStatsDF.append(df)


    ind += 1


fullStatsDF = pd.concat(fullStatsDF, axis=0)
fullStatsDF.to_csv(resultsFile, index=False)

#fullStatsDF.append(df)

if(drawBoxplotGraphs):
    ind = 0
    cols=['Pearson IC50', 'Spearman IC50', 'R2 IC50', 'MSE IC50', 'Pearson Emax',  'Spearman Emax', 'R2 Emax', 'MSE Emax']

    for matrix in [pearsonIC50Matrix, rhoIC50Matrix, r2IC50Matrix, mseIC50Matrix, pearsonEmaxMatrix,  rhoEmaxMatrix, r2EmaxMatrix, mseEmaxMatrix]:

        resultName = cols[ind]
        ind+=1 
        matplotlib.rc('font', size=25)
        matplotlib.rc('axes', titlesize=25)

        ax = sns.boxplot(data=matrix, color='#998FC7')
        ax = sns.swarmplot(data=matrix, color='#14248A')

        ax.set(yticks=np.linspace(0, 1, 11))

        correctedResultName = resultName.replace('IC50', '$ΔIC_{50}$')
        correctedResultName = correctedResultName.replace('Emax', '$ΔE_{max}$')
        correctedResultName = correctedResultName.replace('R2', '$R^2$')

        plt.xlabel("Omics Type", size=30, labelpad=20)
        plt.title('RF Model Performance', size=40)
        plt.ylabel(correctedResultName, size=30, labelpad=7)

        ax.set(xticklabels=['CRISPR-KO','Transcriptomics','Proteomics'])




        figure = plt.gcf()
        figure.set_size_inches(plotXSize, plotYSize)
        fileName = resultName + 'boxplot.png'
        plt.savefig(graphFolder / fileName)
        plt.close()

