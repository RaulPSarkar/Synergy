import scipy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np


def barplot(df, saveFolder, type='gdsc', resultName = 'name', groupedBars=False):

    matplotlib.rc('font', size=30)
    matplotlib.rc('axes', titlesize=30)

    if(type=='almanac'):
        splot=sns.barplot(x='model',y='r2_score',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])
        plt.xlabel("Model Used", size=36)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=36,label_type='center')
        plt.ylabel("Comboscores R\u00b2", size=36)
        plt.show()
        splot=sns.barplot(x='model',y='spearman',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])
        plt.xlabel("Model Used", size=36)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=36,label_type='center')
        plt.ylabel("Comboscores Spearman's rho", size=36)
        plt.show()
        splot=sns.barplot(x='model',y='mean_squared_error',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=36)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=36,label_type='center')
        plt.ylabel("Comboscores MSE", size=36)
        plt.show()

    elif(type=='gdsc'):

        if(groupedBars):
            splot=sns.barplot(x='mainModel', hue='type', y=resultName,data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        else:
            ax1 = plt.subplot(1, 2, 1)
            splot=sns.barplot(x='name',y=resultName,data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        
        #pearson/spearman/r2 graphs go from 0 to 1, but not mse
        if( ('Pearson' in resultName) or ('Spearman' in resultName) or ('R2' in resultName)):
            splot.set(yticks=np.linspace(0, 1, 11))

        plt.xlabel("Model Used", size=36)
        plt.title('Performance for each model', size=54)
        plt.ylabel(resultName, size=36)
        if(not groupedBars):
            plt.bar_label(splot.containers[0], size=36,label_type='center')
            patches = [matplotlib.patches.Patch(color=sns.color_palette(['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])[i], label=t) for i,t in enumerate(t.get_text() for t in splot.get_xticklabels())]
            ax1.axes.get_xaxis().set_visible(False)
            ax2 = plt.subplot(122)
            ax2.set_axis_off()
            ax2.legend(handles=patches, loc='center left')    
        else:
            for i in splot.containers:
                splot.bar_label(i,)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
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


def stackedbarplot(df, saveGraphsFolder, metricName):
    
    matplotlib.rc('font', size=25)
    matplotlib.rc('axes', titlesize=25)
    plt.xlabel("Model", size=36)
    plt.title('Model Performance', size=44)
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



def stackedGroupedbarplot(df, saveGraphsFolder, metricName):
    
    colors = plt.cm.Paired.colors

    fig, ax = plt.subplots()
    (df['Cell CV']+df['Drug CV']+df['Regular CV']).plot(kind='bar', color=[colors[1], colors[0]], rot=0, ax=ax)
    (df['Drug CV']+df['Regular CV']).plot(kind='bar', color=[colors[3], colors[2]], rot=0, ax=ax)
    df['Regular CV'].plot(kind='bar', color=[colors[5], colors[4]], rot=0, ax=ax)
    plt.tight_layout()
    plt.show()


