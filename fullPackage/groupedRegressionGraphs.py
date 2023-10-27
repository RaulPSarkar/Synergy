
import pandas as pd
import statsmodels.api as sm 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy
from pathlib import Path
import os



##########################
##########################
#Change
groupingType = 'pair' #lib (library drug only), pair (drug pairs)
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun115regularplusSingleplusCoeffsplusCType.csv']
predictionNames = ['R2']
graphsFolder =  Path(__file__).parent / 'graphs' / 'groupedRegression'
resultsFolder =  Path(__file__).parent / 'results' / 'drugPairCorrelations'

topXGraphs = 10 #How many top x best/worst library drug graphs to generate
##########################
##########################


if not os.path.exists(graphsFolder/ 'IC50'):
    os.mkdir(graphsFolder / 'IC50')
if not os.path.exists(graphsFolder/ 'Emax'):
    os.mkdir(graphsFolder / 'Emax')
if not os.path.exists(resultsFolder):
    os.mkdir(resultsFolder)



#adapted from https://stackoverflow.com/questions/49895000/regression-by-group-in-python-pandas
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    outputA = result.rsquared

    return result.rsquared



counter = 0

whiskerData = []
whiskerDataEmax = []

for j in predictionPaths:

    df = pd.read_csv(j)

    modelName = predictionNames[counter]


    counter += 1

    df[['Cellname', 'Library', 'Anchor']]

    x = pd.DataFrame()
    if(groupingType=='lib'):
        x = df.groupby('Library').apply(regress, 'y_predIC', ['y_trueIC'])
    elif(groupingType=='pair'):
        x = df.groupby(['Library', 'Anchor']).apply(regress, 'y_predIC', ['y_trueIC'])
    x.name = modelName
    whiskerData.append(x)
    
    topLibrary = x.sort_values( ascending=False)
    outputFile = resultsFolder / 'drugPairsR2-IC50.csv'
    topLibrary.to_csv(outputFile)
    print(topLibrary)
    print(topLibrary.index)
    print(topLibrary.index[0])
    print(topLibrary.index[0][1])

    xEmax = pd.DataFrame()

    if(groupingType=='lib'):
        xEmax = df.groupby('Library').apply(regress, 'y_predEmax', ['y_trueEmax'])
    elif(groupingType=='pair'):
        xEmax = df.groupby(['Library', 'Anchor']).apply(regress, 'y_predEmax', ['y_trueEmax'])

    xEmax.name = modelName
    whiskerDataEmax.append(xEmax)
    
    topLibraryEmax = xEmax.sort_values( ascending=False)
    outputFile = resultsFolder / 'drugPairsR2-Emax.csv'
    topLibraryEmax.to_csv(outputFile)



    for pos in ['best','worst']:

        for i in range(topXGraphs):

            index = i
            if(pos=='worst'):
                index=-i-1
            
            if(groupingType=='lib'):
                selectedDF = df.loc[df['Library'] == topLibrary.index[index] ]
            elif(groupingType=='pair'):
                selectedDF = df.loc[df['Library'] == topLibrary.index[index][0] ]
                selectedDF = selectedDF.loc[df['Anchor'] == topLibrary.index[index][1] ]


            plot = sns.regplot(data=selectedDF, x="y_trueIC", y="y_predIC", scatter_kws={"color": "grey"}, line_kws={"color": "orange"})
            slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                            y=plot.get_lines()[0].get_ydata())
            
            plot.axes.axline((0, 0), (0.01, 0.01), linewidth=4, ls="--", c=".3", color='r')
            plt.axhline(y=0, linewidth=2, c=".3", linestyle='-')
            plt.axvline(x=0, linewidth=1, c=".3", linestyle='-')
            plt.xlabel('True ΔIC50', size=36)
            plt.ylabel('Predicted ΔIC50', size=36)
            if(groupingType=='lib'):
                plt.title(modelName + ' Predictions using ' + topLibrary.index[index] + ' as Library Drug', size=54)
            elif(groupingType=='pair'):
                plt.title(modelName + ' Predictions using ' + topLibrary.index[index][0] + ' as Library and ' + topLibrary.index[index][1] + ' as Anchor Drug', size=54)

            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=40)
            ax.tick_params(axis='y', labelsize=40)

            #to save in fullscreen
            figure = plt.gcf()
            figure.set_size_inches(32, 18)

            fileName = modelName + pos + 'regression' + str(i) + '.png'
            plt.savefig(graphsFolder / 'IC50' / fileName)
            plt.close()

            if(groupingType=='lib'):
                selectedDF = df.loc[df['Library'] == topLibrary.index[index] ]
            elif(groupingType=='pair'):
                selectedDF = df.loc[df['Library'] == topLibrary.index[index][0] ]
                selectedDF = selectedDF.loc[df['Anchor'] == topLibrary.index[index][1] ]

            if(groupingType=='lib'):
                selectedDF = df.loc[df['Library'] == topLibraryEmax.index[index] ]
            elif(groupingType=='pair'):
                selectedDF = df.loc[df['Library'] == topLibraryEmax.index[index][0] ] #select only rows with library
                selectedDF = selectedDF.loc[df['Anchor'] == topLibraryEmax.index[index][1] ] #and then select only rows with anchor


            plot = sns.regplot(data=selectedDF, x="y_trueEmax", y="y_predEmax", scatter_kws={"color": "grey"}, line_kws={"color": "orange"})
            slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                            y=plot.get_lines()[0].get_ydata())
            
            plot.axes.axline((0, 0), (0.01, 0.01), linewidth=4, ls="--", c=".3", color='r')
            plt.axhline(y=0, linewidth=2, c=".3", linestyle='-')
            plt.axvline(x=0, linewidth=1, c=".3", linestyle='-')
            plt.xlabel('True ΔEmax', size=36)
            plt.ylabel('Predicted ΔEmax', size=36)
            if(groupingType=='lib'):
                plt.title(modelName + ' Predictions using ' + topLibraryEmax.index[index] + ' as Library Drug', size=54)
            elif(groupingType=='pair'):
                plt.title(modelName + ' Predictions using ' + topLibraryEmax.index[index][0] + ' as Library and ' + topLibraryEmax.index[index][1] + ' as Anchor Drug', size=54)


            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=40)
            ax.tick_params(axis='y', labelsize=40)
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)

            fileName = modelName + pos + 'regression' + str(i) + '.png'
            plt.savefig(graphsFolder / 'Emax' / fileName)
            plt.close()



ax = sns.boxplot(data=whiskerData)
plt.xlabel("Models", size=46)
plt.title('Box Plot of grouped correlations by library drugs', size=54)
plt.ylabel("R\u00b2 of ΔIC50", size=36)
ax.set_xticklabels(predictionNames, fontsize=36)
ax.tick_params(axis='y', labelsize=36)

figure = plt.gcf()
figure.set_size_inches(32, 18)
plt.savefig(graphsFolder / 'IC50' / 'boxplot.png')
plt.close()



axEmax = sns.boxplot(data=whiskerDataEmax)
plt.xlabel("Models", size=46)
plt.title('Box Plot of grouped correlations by library drugs', size=54)
plt.ylabel("R\u00b2 of ΔEmax", size=36)
axEmax.set_xticklabels(predictionNames, fontsize=36)
axEmax.tick_params(axis='y', labelsize=36)

figure = plt.gcf()
figure.set_size_inches(32, 18)
plt.savefig(graphsFolder / 'Emax' / 'boxplot.png')

print("All graphs saved!")