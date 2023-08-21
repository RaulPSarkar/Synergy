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
shapValuesDF = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesEmax.parquet.gzip'
localSelection = ['A', 'B'] #if wanting local (only SHAP for a certain drug pair)
numberOfTopFeatures = 20 #number of most important features to select

##########################
##########################

shapValuesDF = pd.read_parquet(shapValuesDF)

#ass = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesIC50.parquet.gzip'
#shapValuesDF.to_parquet(ass, compression='gzip')  
#shapValuesDF.to_csv(ass)


print(shapValuesDF.head(5))


globalValues = shapValuesDF.abs().mean(axis=1)
topFeatures = globalValues.sort_values( ascending=False).head(numberOfTopFeatures)
#top 10 most important featurees, as determined by mean absolute SHAP
print(topFeatures)
