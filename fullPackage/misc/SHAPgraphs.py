import pandas as pd
from pathlib import Path
import os
#####################################################
#WARNING: YOU NEED TO "pip install fastparquet" TO RUN THIS!
#####################################################

#Note: Feature importance is calculated from mean absolute SHAP

##########################
##########################

predictionsDF = Path(__file__).parent / 'predictions/final/lgbm/lgbmrun115regularplusSingleplusCoeffsplusCType.csv' #very important to determine samples used
shapValuesIC50 = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesIC50.parquet.gzip'
shapValuesEmax = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesEmax.parquet.gzip'
filterBy = ['Tissue'] #if wanting local (only SHAP for a certain drug pair/tissue)
joinIC50andEmax = False #whether to join these SHAP values onto a single value (useless for now)
numberOfTopFeatures = 20 #number of most important features to select

##########################
##########################


def filterRowsByProperty(property, predictionsDF):
    pass

predictionsDF = pd.read_csv(predictionsDF, index_col=0)
shapValuesIC50 = pd.read_parquet(shapValuesIC50)
shapValuesEmax = pd.read_parquet(shapValuesEmax)

shapValuesIC50.columns = predictionsDF.index
shapValuesEmax.columns = predictionsDF.index

print(shapValuesIC50)

globalValues = shapValuesIC50.abs().mean(axis=1)
topFeatures = globalValues.sort_values( ascending=False).head(numberOfTopFeatures)
#top 10 most important featurees, as determined by mean absolute SHAP
print(topFeatures)

print("Breast")

