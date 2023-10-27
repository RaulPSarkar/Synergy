import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#####################################################
#WARNING: YOU NEED TO "pip install fastparquet" TO RUN THIS!
#####################################################

#Note: Feature importance is calculated from mean absolute SHAP
#Note: Genes with suffix 'A' are from library drug, 'B' are from anchor drug

##########################
##########################
plotXSize = 16 #x and y sizes to render the graphs on (smaller x makes bars thinner)
plotYSize = 16



#predictionsDF = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun116cellplusSingleplusCoeffsplusCType.csv' #very important to determine samples used
#shapValuesIC50 = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun116SHAPvaluesIC50.parquet.gzip'
#shapValuesEmax = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun116SHAPvaluesEmax.parquet.gzip'

predictionsDF = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115regularplusSingleplusCoeffsplusCType.csv' #very important to determine samples used
shapValuesIC50 = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesIC50.parquet.gzip'
shapValuesEmax = Path(__file__).parent / '../predictions/final/lgbm/lgbmrun115SHAPvaluesEmax.parquet.gzip'

saveGraphsFolder =  Path(__file__).parent / '../graphs' / 'SHAP'

filterColumn = 'Library' #i.e. "Tissue"
filter = 'Camptothecin' #i.e. "Breast" (only select values of "Breast" for column name "Tissue")
useFilter = False #whether to use the filter above

secondfilterColumn = 'Anchor' #i.e. "Tissue"
secondFilter = 'Pictilisib' #i.e. "Breast" (only select values of "Breast" for column name "Tissue")
useSecondFilter = False #whether to use the filter above

performIC50Shap = False #if false, it will do SHAP of Emax values instead
#joinIC50andEmax = False #whether to join these SHAP values onto a single value (useless for now)
numberOfTopFeatures = 20 #number of most important features to select

filterOutSingleAgent = True #
filterOutCancerType = True #

####
makeAltGraphs = True #keep it false. Used to make the table on mean grouped SHAP value (single agents/expression/coefficients), and corresponding histograms
altTableFile = Path(__file__).parent / '../results/SHAP/groupedFeaturesTableEmax.csv'
histogramFile = Path(__file__).parent / '../graphs/SHAP/'
####

fileName = 'SHAPtempIgnore' #output file name
plotTitle = 'Highest Absolute Gene SHAP Values for Library=Camptothecin, Anchor=Pictilisib'
yLabelText = 'ΔIC50 SHAP Value'
##########################
##########################



if(useFilter):
    fileName += filter
if(useSecondFilter):
    fileName += secondFilter
fileName += '.png'

def filterRowsByProperty(filterColumn, filter, predictionsDF, shapDF, secondFilter=None, secondFilterColumn=None):
    if(secondFilter==None):
        columnNames = predictionsDF.index[predictionsDF[filterColumn] == filter].tolist() #select the indexes through supplementary predictions file
    else:
        columnNames = predictionsDF.index[   (predictionsDF[filterColumn] == filter) & (predictionsDF[secondFilterColumn] == secondFilter)      ].tolist() #select the indexes through supplementary predictions file


    return shapDF[columnNames]


def removeFeature(shapDF, featureName):
    shapDF.drop(featureName, inplace=True)
    return shapDF

def selectTopNFeatures(shapDF, n=0):
    globalValues = shapDF.abs().mean(axis=1)
    if(n>0):
        topFeatures = globalValues.sort_values( ascending=False).head(n)
    else:
        topFeatures = globalValues.sort_values( ascending=False)

    return topFeatures

def groupedFeatureMeans(df):
    columnsSum = df.abs().sum(axis=0)
    return columnsSum.mean()

def groupedFeatureMedian(df):
    columnsSum = df.abs().sum(axis=0)
    return columnsSum.median()


predictionsDF = pd.read_csv(predictionsDF, index_col=0)
shapValuesIC50 = pd.read_parquet(shapValuesIC50)
shapValuesEmax = pd.read_parquet(shapValuesEmax)
shapValuesIC50.columns = predictionsDF.index
shapValuesEmax.columns = predictionsDF.index


if(not performIC50Shap):
    shapValuesIC50 = shapValuesEmax #this is the lazy way to do it

if(makeAltGraphs):
    singleAgentIC50 = shapValuesIC50.loc[['Library IC50','Library Emax','Anchor Emax','Anchor IC50']]
    cancerTypeIC50 = shapValuesIC50.loc[['cancerTypeFactorized']]

if(filterOutSingleAgent):
    for singleAgent in ['Library IC50','Library Emax','Anchor Emax','Anchor IC50']:
        shapValuesIC50 = removeFeature(shapValuesIC50, singleAgent) #beautiful

if(filterOutCancerType):
    shapValuesIC50 = removeFeature(shapValuesIC50, 'cancerTypeFactorized') #beautiful


if(makeAltGraphs):
    drugAValuesIC50 = shapValuesIC50[shapValuesIC50.index.str.contains(' A')]
    drugBValuesIC50 = shapValuesIC50[shapValuesIC50.index.str.contains(' B')]
    geValuesIC50 = shapValuesIC50[~shapValuesIC50.index.str.contains(' A| B')]


    groupsArray = ['Expression', 'Coefficient A', 'Coefficient B'] 
    index = 0
    for histDF in [geValuesIC50, drugAValuesIC50, drugBValuesIC50]: #makes histogram graphs
        group = groupsArray[index]
        index+=1
        columnsMax = histDF.abs().max(axis=0)
        matplotlib.rc('font', size=30)
        matplotlib.rc('axes', titlesize=30)
        plt.ylabel('Count', size=26, labelpad=16)
        plt.title('Histogram of highest ' + group +' SHAP value per observation', size=30, pad = 18)
        plt.xlabel('Highest $ΔE_{max}$ SHAP Value in each Observation', size=26, labelpad=16)
        plot = sns.histplot(data=columnsMax)
        figure = plt.gcf()
        figure.set_size_inches(14, 16)
        name = 'histEmax' + group + '.png'
        histName = histogramFile / name
        plt.xlim([0, 0.045])
        plt.savefig(histName)
        plt.close()


    smallTableData = [['Single Agent', groupedFeatureMeans(singleAgentIC50 ), groupedFeatureMedian(singleAgentIC50)], ['Drug A Coeffs.', groupedFeatureMeans(drugAValuesIC50 ), groupedFeatureMedian(drugAValuesIC50)], ['Drug B Coeffs.',groupedFeatureMeans(drugBValuesIC50 ), groupedFeatureMedian(drugBValuesIC50)], ['Gene Expression',groupedFeatureMeans(geValuesIC50 ), groupedFeatureMedian(geValuesIC50)]] 
    smallTable = pd.DataFrame(smallTableData, columns=['Input', 'Mean Absolute SHAP for Grouped Features', 'Median Absolute SHAP for Grouped Features'])
    
    print(smallTable)
    smallTable.to_csv(altTableFile, index=False)


if(useFilter and useSecondFilter):

    shapFiltered = filterRowsByProperty(filterColumn, filter, predictionsDF, shapValuesIC50, secondFilter=secondFilter, secondFilterColumn=secondfilterColumn)
    topFeatures = selectTopNFeatures(shapFiltered, numberOfTopFeatures)

elif(useSecondFilter):
    shapFiltered = filterRowsByProperty(secondfilterColumn, secondFilter, predictionsDF, shapValuesIC50)
    topFeatures = selectTopNFeatures(shapFiltered, numberOfTopFeatures)
else:
    topFeatures = selectTopNFeatures(shapValuesIC50, numberOfTopFeatures)



matplotlib.rc('font', size=22)
matplotlib.rc('axes', titlesize=22)
fig, ax = plt.subplots()

sns.barplot(x=topFeatures.index, y=topFeatures.values)
figure = plt.gcf()
figure.set_size_inches(plotXSize, plotYSize)

if not os.path.exists(saveGraphsFolder):
    os.mkdir(saveGraphsFolder)


plt.ylabel(yLabelText, size=26, labelpad= 2)
plt.title(plotTitle, size=28)
#plt.title('Features with Highest Absolute SHAP Values in Breast', size=28)
plt.xlabel("Feature", size=26, labelpad=0)

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


plt.savefig(saveGraphsFolder / fileName)
plt.close()

print(topFeatures)
