import pandas as pd
from pathlib import Path
import os


#Feature importance is calculated from mean absolute SHAP

##########################
##########################

predictionsDF = Path(__file__).parent / 'predictions/final/lgbm/lgbmrun115regularplusSingleplusCoeffsplusCType.csv' #very important to determine samples used
shapValuesDF = Path(__file__).parent / '../predictions/temp/lgbmshap0ic.csv'
localSelection = ['A', 'B'] #if wanting local (only SHAP for a certain drug pair)
numberOfTopFeatures = 20 #number of most important features to select

##########################
##########################

shapValuesDF = pd.read_csv(shapValuesDF, index_col=0)
shapValuesDF = shapValuesDF.head(200)

print(shapValuesDF.head(5))


globalValues = shapValuesDF.abs().mean(axis=1)
topFeatures = globalValues.sort_values( ascending=False).head(numberOfTopFeatures)

print(topFeatures)
