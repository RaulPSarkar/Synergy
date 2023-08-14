import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

######################
######################
makeGraphsInstead = False
graphsFolder =  Path(__file__).parent / 'graphs' / 'coeffsThreshold'
processWithThresholdsBelow = True #whether to do any processing of which genes to keep (with params below)
minimumNonNull = 20 #threshold of minimum amount of non-null coefficients for gene to be maintained
coefficientValueThreshold = 0.02
minimumNumberAbove = 0#3 #the minimum number of coefficients that should be over the threshold above (i.e. 3 coefficients should be above 0.01 for the gene to be maintained)
coefficientValueSecondThreshold = 0.08 #a second threshold, similar to the previous one
minimumSecondNumberAbove = 0#1 #the minimum number of coefficients that should be over the threshold above (i.e. 3 coefficients should be above 0.01 for the gene to be maintained)
coeffs = Path(__file__).parent / 'datasets/CoefficientsLasso.csv'
outputFile = Path(__file__).parent / "datasets/coefsProcessedWithThreshold.csv"
######################
######################




coeffs = pd.read_csv(coeffs, index_col=0)
#drop all the intercepts (not needed, for now)
coeffs = coeffs[coeffs['features'] != '(Intercept)']
coeffs = coeffs.drop(['Drug_id'], axis=1)
coeffs = coeffs.drop_duplicates(subset=['Drug_name','features'], keep='last')
df = pd.pivot(coeffs, index='features', columns='Drug_name', values='EN_coef')


#####ADDING THRESHOLDS TO REDUCE TOTAL NUMBER OF GENES USED TO TRAIN THE MODEL
##############################################################################

if(makeGraphsInstead):
    df.fillna(0, inplace=True)
    #nonNullAmounts = df.astype(bool).sum(axis=1)
    #firstFiltered = df.loc[nonNullAmounts >= minimumNonNull] #just selecting rows above threshold (based on nonNullAmounts series)
    aboveThresholdFrequency = df[abs(df) >= coefficientValueThreshold].count(axis=1) #needs to be the absolute value
    secondFiltered = df.loc[aboveThresholdFrequency >= minimumNumberAbove]

    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)
    plot = sns.histplot(data=aboveThresholdFrequency)
    plt.xlim([0,50])
    plot.set(xticks=np.linspace(0, 50, 11))

    figure = plt.gcf()
    figure.set_size_inches(32, 18)
    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)
    plt.xlabel("Number of coefficients above " + str(coefficientValueThreshold) +  " in gene", size=36)
    plt.axline((minimumNumberAbove-0.5, 0), (minimumNumberAbove-0.5, 1), linestyle='dashed', color='red', linewidth=5)
    plot.text(minimumNumberAbove, 3000, str(minimumNumberAbove) + '+ coefficients >=' + str(coefficientValueThreshold), color='red')
    plt.title('Histogram of number of coefficients above ' + str(coefficientValueThreshold) + ' present in a gene\n (i.e. how many drugs are at least minimally dependant on that gene)', size=44)
    plt.ylabel('Amount of genes', size=36)

    fileName = 'threshold1' + '.png'
    plt.savefig(graphsFolder / fileName)
    plt.close()



    aboveSecondThresholdFrequency = df[abs(df) >= coefficientValueSecondThreshold].count(axis=1) #needs to be the absolute value
    thirdFiltered = df.loc[aboveSecondThresholdFrequency >= minimumSecondNumberAbove]


    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)
    plot = sns.histplot(data=aboveSecondThresholdFrequency, bins=100)
    plt.xlim([0,10])
    plot.set(xticks=np.linspace(0, 10, 11))

    figure = plt.gcf()
    figure.set_size_inches(32, 18)
    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)
    plt.xlabel("Number of coefficients above " + str(coefficientValueSecondThreshold) +  " in gene", size=36)
    plt.axline((minimumSecondNumberAbove, 0), (minimumSecondNumberAbove, 1), linestyle='dashed', color='red', linewidth=5)
    plot.text(minimumSecondNumberAbove+0.5, 10000, str(minimumSecondNumberAbove) + '+ coefficients >=' + str(coefficientValueSecondThreshold), color='red')
    plt.title('Histogram of number of coefficients above ' + str(coefficientValueSecondThreshold) + ' present in a gene\n (i.e. how many drugs are at least minimally dependant on that gene)', size=44)
    plt.ylabel('Amount of genes', size=36)

    fileName = 'threshold2' + '.png'
    plt.savefig(graphsFolder / fileName)
    plt.close()

else:
    if(processWithThresholdsBelow):
        df.fillna(0, inplace=True)
        #TAKEN FROM https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
        nonNullAmounts = df.astype(bool).sum(axis=1)
        firstFiltered = df.loc[nonNullAmounts >= minimumNonNull] #just selecting rows above threshold (based on nonNullAmounts series)

        aboveThresholdFrequency = firstFiltered[abs(firstFiltered) >= coefficientValueThreshold].count(axis=1) #needs to be the absolute value
        secondFiltered = firstFiltered.loc[aboveThresholdFrequency >= minimumNumberAbove]


        aboveSecondThresholdFrequency = secondFiltered[abs(secondFiltered) >= coefficientValueSecondThreshold].count(axis=1) #needs to be the absolute value
        thirdFiltered = secondFiltered.loc[aboveSecondThresholdFrequency >= minimumSecondNumberAbove]
        thirdFiltered.to_csv(outputFile)


        
        thirdFilteredSparse = thirdFiltered.replace(0, np.nan)
        thirdFilteredSparse = thirdFilteredSparse.apply(pd.arrays.SparseArray)
        dfSparse = df.replace(0, np.nan)
        dfSparse = dfSparse.apply(pd.arrays.SparseArray)

        
        print ('\nOriginal Values')
        print('Mean (non-null values): ' + str(dfSparse.mean(skipna=True).mean() ) )
        print ('Density: ' + str(dfSparse.sparse.density))
        print("Number of genes: " + str(dfSparse.shape[0])  + '\n' )

        print ('Post-Processing Values')
        print('Mean (non-null values): ' + str(thirdFilteredSparse.mean(skipna=True).mean() ) )
        print ('Density: ' + str(thirdFilteredSparse.sparse.density) )
        print("Number of genes: " + str(thirdFilteredSparse.shape[0]))


    else:
        df.to_csv(outputFile)






##############################################################################


