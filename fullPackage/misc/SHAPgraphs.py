import pandas as pd
from pathlib import Path
import os


#Feature importance is calculated from mean absolute SHAP

##########################
##########################

predictionsDF = Path(__file__).parent / 'predictions/final/lgbm/lgbmrun115.csv' #very important to determine samples used
shapValuesDF = Path(__file__).parent / 'predictions/final/lgbm/lgbmrun115.csv'
localSelection = ['A', 'B'] #if wanting local (only SHAP for a certain drug pair)
numberOfTopFeatures = 20 #number of most important features to select

##########################
##########################

shapValuesDF = pd.read_csv(shapValuesDF)
print(shapValuesDF.head(5))


globalValues = shapValuesDF.abs().mean(axis=1)
topFeatures = globalValues.sort_values( ascending=False).head(numberOfTopFeatures)

print(topFeatures)
