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
#Regular Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DLrun1.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensemble.csv'




#Coeffs Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99regularplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs'/ 'dlCoeffsrun111regulargeplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun114regulargeplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun115regulargeplusCoeffs.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleCoeffs.csv'



#Single+Coeffs Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun100regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs'/ 'dlCoeffsrun111regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun100regularplusSingleplusCoeffs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun115regularplusSingleplusCoeffs.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleSinglePlusCoeffs.csv'


#Single+Coeffs+Type Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs'/ 'dlCoeffsrun111regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun115regularplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun115regularplusSingleplusCoeffsplusCType.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleSinglePlusCoeffsPlusType.csv'


#Shuffled Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun33.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'DLrun6.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun33.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun33.csv']
#predictionWeights = [1,1,1,1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleShuffled.csv'


#Drug CV Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'dlrun1drug.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11drugplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99drugplusDrugs.csv']
#predictionWeights = [1,1,1,1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleDrug.csv'


#Single+Coeffs+Type Drug CV Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun117drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs'/ 'dlCoeffsrun111drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun115drugplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun115drugplusSingleplusCoeffsplusCType.csv']
#predictionWeights = [1,1,1, 1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleDrugSinglePlusCoeffsPlusType.csv'



#Cell CV Ensemble
#predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun99cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'DL'/ 'dlrun1cell.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun11cellplusDrugs.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun99cellplusDrugs.csv']
#predictionWeights = [1,1,1,1]
#outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleCell.csv'

#Single+Coeffs+Type Cell CV Ensemble
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun116cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'dlCoeffs'/ 'dlCoeffsrun111cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'rf'/ 'rfrun115cellplusSingleplusCoeffsplusCType.csv', Path(__file__).parent / 'predictions' /'final'/'xgboost'/ 'xgboostrun115cellplusSingleplusCoeffsplusCType.csv']
predictionWeights = [1,1,1, 1]
outputEnsembleFile = Path(__file__).parent / 'predictions'  /'final'/ 'ensemble' / 'ensembleCellSinglePlusCoeffsPlusType.csv'





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