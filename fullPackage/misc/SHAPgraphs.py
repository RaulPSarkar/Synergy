import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt

#####################################################
#WARNING: YOU NEED TO "pip install fastparquet" TO RUN THIS!
#####################################################

#Note: Feature importance is calculated from mean absolute SHAP

##########################
##########################

predictionsDF = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun116cellplusSingleplusCoeffsplusCType.csv' #very important to determine samples used
shapValuesIC50 = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun116SHAPvaluesIC50.parquet.gzip'
shapValuesEmax = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun116SHAPvaluesEmax.parquet.gzip'

#shapValuesIC50 = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesIC50.parquet.gzip'
#shapValuesEmax = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesEmax.parquet.gzip'

saveGraphsFolder =  Path(__file__).parent / '../graphs' / 'SHAP'

filterColumn = 'Tissue' #i.e. "Tissue"
filter = 'Breast' #i.e. "Breast" (only select values of "Breast" for column name "Tissue")
useFilter = False #whether to use the filter above
joinIC50andEmax = False #whether to join these SHAP values onto a single value (useless for now)
numberOfTopFeatures = 10 #number of most important features to select

##########################
##########################



def filterRowsByProperty(filterColumn, filter, predictionsDF, shapDF):
    columnNames = predictionsDF.index[predictionsDF[filterColumn] == filter].tolist() #select the indexes through supplementary predictions file
    return shapDF[columnNames]

def selectTopNFeatures(shapDF, n=10):
    globalValues = shapDF.abs().mean(axis=1)
    topFeatures = globalValues.sort_values( ascending=False).head(n)
    return topFeatures

predictionsDF = pd.read_csv(predictionsDF, index_col=0)
shapValuesIC50 = pd.read_parquet(shapValuesIC50)
shapValuesEmax = pd.read_parquet(shapValuesEmax)

shapValuesIC50.columns = predictionsDF.index
shapValuesEmax.columns = predictionsDF.index

print(shapValuesIC50)


if(useFilter):
    shapFiltered = filterRowsByProperty(filterColumn, filter, predictionsDF, shapValuesIC50)
    topFeatures = selectTopNFeatures(shapFiltered, numberOfTopFeatures)
else:
    topFeatures = selectTopNFeatures(shapValuesIC50, numberOfTopFeatures)


sns.barplot(x=topFeatures.index, y=topFeatures.values)
figure = plt.gcf()
figure.set_size_inches(32, 18)

fileName = 'globalSHAP2.png'
if not os.path.exists(saveGraphsFolder):
    os.mkdir(saveGraphsFolder)
plt.savefig(saveGraphsFolder / fileName)
plt.close()

print(topFeatures)
