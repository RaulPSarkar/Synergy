import scipy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np


def barplot(df, saveFolder, type='gdsc', resultName = 'name', groupedBars=False, barplotXSize = 32, barplotYSize = 18, drawBarPlotValues=False, groupedBarplotLegendLocation='lower center', tiltAngle=0, title= 'Performance for each model', titleOffset=6):

    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)

    
    fig, ax = plt.subplots()

    if(groupedBars):
        splot=sns.barplot(x='mainModel', hue='type', y=resultName,data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        splot.legend(loc=groupedBarplotLegendLocation)

    else:
        ax1 = plt.subplot(1, 2, 1)
        splot=sns.barplot(x='name',y=resultName,data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])

    #pearson/spearman/r2 graphs go from 0 to 1, but not mse
    if( ('Pearson' in resultName) or ('Spearman' in resultName) or ('R2' in resultName)):
        splot.set(yticks=np.linspace(0, 1, 11))

    correctedResultName = resultName.replace('IC50', '$ΔIC_{50}$')
    correctedResultName = correctedResultName.replace('Emax', '$ΔE_{max}$')
    correctedResultName = correctedResultName.replace('R2', '$R^2$')

    if(type=='almanac'):
        correctedResultName = correctedResultName + ' COMBOSCOREs'

    plt.xlabel("Model Used", size=30)
    plt.title(title, size=40, pad=titleOffset)
    plt.ylabel(correctedResultName, size=30)
    if(not groupedBars):
        #plt.bar_label(splot.containers[0], size=30,label_type='center')
        patches = [matplotlib.patches.Patch(color=sns.color_palette(['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])[i], label=t) for i,t in enumerate(t.get_text() for t in splot.get_xticklabels())]
        ax1.axes.get_xaxis().set_visible(False)
        ax2 = plt.subplot(122)
        ax2.set_axis_off()
        ax2.legend(handles=patches, loc='center left')    
    else:
        if(drawBarPlotValues):
            for i in splot.containers:
                    splot.bar_label(i,)

        plt.setp(ax.get_xticklabels(), rotation=tiltAngle, horizontalalignment='right')
        

    #to save in fullscreen
    figure = plt.gcf()
    figure.set_size_inches(barplotXSize, barplotYSize)
    fileName = resultName + 'barplot.png'
    if not os.path.exists(saveFolder / 'barPlots'):
        os.mkdir(saveFolder / 'barPlots')
    plt.savefig(saveFolder / 'barPlots' / fileName)
    plt.close()


def regressionGraphs(df, modelName, dfStats, saveGraphsFolder):

    textIC = "Spearman's rho: " + str ( dfStats.loc[modelName]['Spearman IC50'] ) + "\n" + "R\u00b2: " + str ( dfStats.loc[modelName]['R2 IC50'] ) + "\n" + "MSE: " + str ( dfStats.loc[modelName]['MSE IC50'] )
    textEmax = "Spearman's rho: " + str ( dfStats.loc[modelName]['Spearman Emax'] ) + "\n" + "R\u00b2: " + str ( dfStats.loc[modelName]['R2 Emax'] ) + "\n" + "MSE: " + str ( dfStats.loc[modelName]['MSE Emax'] )
    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)
    #plt.show()
    plot = sns.regplot(data=df, x="y_trueIC", y="y_predIC", scatter_kws={"color": "grey"}, line_kws={"color": "orange"})#, hue="Tissue")
    slope, intercept, r, p, sterr = scipy.stats.linregress(x=plot.get_lines()[0].get_xdata(),
                                                       y=plot.get_lines()[0].get_ydata())

    plot.axes.axline((0, 0), (0.01, 0.01), linewidth=4, ls="--", c=".3", color='r')
    plt.axhline(y=0, linewidth=2, c=".3", linestyle='-')

    plt.xlabel('True ΔIC50')
    plt.ylabel('Predicted ΔIC50')
    plt.title(modelName +' Model Results')
    ax = plt.gca()
    props = dict(facecolor='grey', alpha=0.3)
    ax.text(0.05, 0.95, textIC, transform=ax.transAxes, fontsize=24,
            verticalalignment='top', bbox=props)

    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)

    #to save in fullscreen
    figure = plt.gcf()
    figure.set_size_inches(32, 18)

    fileName = modelName + 'scatter.png'
    if not os.path.exists(saveGraphsFolder / 'IC50'):
        os.mkdir(saveGraphsFolder / 'IC50')
    plt.savefig(saveGraphsFolder / 'IC50' / fileName)
    plt.close()


    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)
    plot = sns.regplot(data=df, x="y_trueEmax", y="y_predEmax", scatter_kws={"color": "grey"}, line_kws={"color": "orange"})#, hue="Tissue")

    plot.axes.axline((0, 0), (0.01, 0.01), linewidth=4, ls="--", c=".3", color='r')
    plt.axhline(y=0, linewidth=2, c=".3", linestyle='-')

    plt.xlabel('True ΔEmax')
    plt.ylabel('Predicted ΔEmax')
    plt.title(modelName +' Model Results')
    ax = plt.gca()
    props = dict(facecolor='grey', alpha=0.3)
    ax.text(0.05, 0.95, textEmax, transform=ax.transAxes, fontsize=24,
            verticalalignment='top', bbox=props)
    
    #to save in fullscreen
    figure = plt.gcf()
    figure.set_size_inches(32, 18)

    fileName = modelName + 'scatter.png'
    if not os.path.exists(saveGraphsFolder / 'Emax'):
        os.mkdir(saveGraphsFolder / 'Emax')
    plt.savefig(saveGraphsFolder / 'Emax' / fileName)
    plt.close()


def stackedbarplot(df, saveGraphsFolder, metricName):
    
    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)
    plt.xlabel("Model", size=36)
    plt.title('Model Performance', size=44)
    correctedResultName = metricName.replace('IC50', '$ΔIC_{50}$')
    correctedResultName = correctedResultName.replace('Emax', '$ΔE_{max}$')
    correctedResultName = correctedResultName.replace('R2', '$R^2$')

    plt.ylabel(metricName, size=36)

    barlist = df.set_index('Model').plot(kind='bar', stacked=True)
    barlist.set(yticks=np.linspace(0, 1, 11))
    figure = plt.gcf()
    figure.set_size_inches(32, 18)

    fileName = 'stackedBar' + metricName + '.png'
    if not os.path.exists(saveGraphsFolder / 'IC50'):
        os.mkdir(saveGraphsFolder / 'IC50')
    plt.savefig(saveGraphsFolder / 'IC50' / fileName)
    plt.close()



def stackedGroupedbarplot(df, saveGraphsFolder, metricName, barplotXSize = 16, barplotYSize = 18, groupedBarplotLegendLocation='lower center'):
    
    colors = plt.cm.Paired.colors
    matplotlib.rc('font', size=22)
    matplotlib.rc('axes', titlesize=22)

    fig, ax = plt.subplots()
    (df['Regular CV']+df['Drug CV']+df['Cell CV']).plot(kind='bar', color=[colors[1], colors[0]], rot=0, ax=ax)
    (df['Drug CV']+df['Cell CV']).plot(kind='bar', color=[colors[3], colors[2]], rot=0, ax=ax)
    df['Cell CV'].plot(kind='bar', color=[colors[5], colors[4]], rot=0, ax=ax)
    figure = plt.gcf()
    figure.set_size_inches(barplotXSize, barplotYSize)
    
    plt.legend(loc=groupedBarplotLegendLocation, ncol=1, fontsize=24, labels=['CV (Coeffs+Single Agent+CType)', 'CV (Fingerprints)', 'Drug Pair CV (Coeffs+Single Agent+CType)', 'Drug Pair CV (Fingerprints)', 'Cell CV (Coeffs+Single Agent+CType)', 'Cell CV (Fingerprints)'])
    
    #plt.legend(['line1', 'line2', 'line3'], ['label1', 'label2', 'label3'])
    #ax.legend(labels=mylabels)

    #pearson/spearman/r2 graphs go from 0 to 1, but not mse
    if( ('Pearson' in metricName) or ('Spearman' in metricName) or ('R2' in metricName)):
        plt.yticks(np.linspace(0, 1, 11))

    correctedResultName = metricName.replace('IC50', '$ΔIC_{50}$')
    correctedResultName = correctedResultName.replace('Emax', '$ΔE_{max}$')
    correctedResultName = correctedResultName.replace('R2', '$R^2$')

    plt.ylabel(correctedResultName, size=26)
    plt.title('Performance of each CV Stratification Method', size=28)
    plt.xlabel("Model Name", size=26)

    #plt.tight_layout()
    #plt.show()
    plt.savefig(saveGraphsFolder / metricName)
    plt.close()


