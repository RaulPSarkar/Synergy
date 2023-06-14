
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
predictionPaths = [Path(__file__).parent / 'predictions' /'final'/'en'/ 'enrun0.csv', Path(__file__).parent / 'predictions' /'final'/'lgbm'/ 'lgbmrun0.csv']#, 'predictions/MultiNoConc/predictionsExpressionRF.csv', 'predictions/MultiNoConc/predictionsMultiExpressionEN.csv', 'predictions/MultiNoConc/predictionsMultiExpressionLGBM.csv', 'predictions/MultiNoConc/predictionsMultiExpressionXGB.csv']
predictionNames = ['EN', 'LGBM']#'Baseline','DL', 'RF', 'EN', 'LGBM', 'XGBoost']
graphsFolder =  Path(__file__).parent / 'graphs' / 'groupedRegression'
topXGraphs = 3 #How many top x best/worst library drug graphs to generate
##########################
##########################


if not os.path.exists(graphsFolder/ 'IC50'):
    os.mkdir(graphsFolder / 'IC50')
if not os.path.exists(graphsFolder/ 'Emax'):
    os.mkdir(graphsFolder / 'Emax')


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
    x = df.groupby('Library').apply(regress, 'y_predIC', ['y_trueIC'])
    x.name = modelName
    whiskerData.append(x)
    
    topLibrary = x.sort_values( ascending=False)


    xEmax = pd.DataFrame()
    xEmax = df.groupby('Library').apply(regress, 'y_predEmax', ['y_trueEmax'])
    xEmax.name = modelName
    whiskerDataEmax.append(xEmax)
    
    topLibraryEmax = xEmax.sort_values( ascending=False)



    for pos in ['best','worst']:

        for i in range(3):

            index = i
            if(pos=='worst'):
                index=-i-1
            
            selectedDF = df.loc[df['Library'] == topLibrary.index[index] ]
            plot = sns.regplot(data=selectedDF, x="y_trueIC", y="y_predIC", scatter_kws={"color": "grey"}, line_kws={"color": "orange"})
            slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                            y=plot.get_lines()[0].get_ydata())
            
            plot.axes.axline((0, 0), (1, 1), linewidth=4, ls="--", c=".3", color='r')
            plt.axhline(y=0, linewidth=2, c=".3", linestyle='-')
            plt.axvline(x=0, linewidth=1, c=".3", linestyle='-')
            plt.xlabel('True ΔIC50', size=36)
            plt.ylabel('Predicted ΔIC50', size=36)
            plt.title(modelName + ' Predictions using ' + topLibrary.index[index] + ' as Library Drug', size=54)
            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=40)
            ax.tick_params(axis='y', labelsize=40)
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)

            fileName = modelName + pos + 'regression' + str(i) + '.png'
            plt.savefig(graphsFolder / 'IC50' / fileName)
            plt.close()






            selectedDF = df.loc[df['Library'] == topLibraryEmax.index[index] ]
            plot = sns.regplot(data=selectedDF, x="y_trueEmax", y="y_predEmax", scatter_kws={"color": "grey"}, line_kws={"color": "orange"})
            slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                            y=plot.get_lines()[0].get_ydata())
            
            plot.axes.axline((0, 0), (1, 1), linewidth=4, ls="--", c=".3", color='r')
            plt.axhline(y=0, linewidth=2, c=".3", linestyle='-')
            plt.axvline(x=0, linewidth=1, c=".3", linestyle='-')
            plt.xlabel('True ΔEmax', size=36)
            plt.ylabel('Predicted ΔEmax', size=36)
            plt.title(modelName + ' Predictions using ' + topLibraryEmax.index[index] + ' as Library Drug', size=54)
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
