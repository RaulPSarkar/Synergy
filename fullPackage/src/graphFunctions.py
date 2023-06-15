import scipy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os



def barplot(df, saveFolder, type='gdsc'):

    matplotlib.rc('font', size=30)
    matplotlib.rc('axes', titlesize=30)

    if(type=='almanac'):
        splot=sns.barplot(x='model',y='r2_score',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("Comboscores R\u00b2", size=26)
        plt.show()
        splot=sns.barplot(x='model',y='spearman',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("Comboscores Spearman's rho", size=26)
        plt.show()
        splot=sns.barplot(x='model',y='mean_squared_error',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("Comboscores MSE", size=26)
        plt.show()

    elif(type=='gdsc'):

        splot =sns.barplot(x='name',y='R2 IC50',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("R\u00b2 ΔIC50", size=26)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        fileName = 'R2IC50barplot.png'
        if not os.path.exists(saveFolder / 'barPlots'):
            os.mkdir(saveFolder / 'barPlots')
        plt.savefig(saveFolder / 'barPlots' / fileName)
        plt.close()




        splot=sns.barplot(x='name',y='R2 Emax',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("R\u00b2 ΔEmax", size=26)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        fileName = 'R2Emaxbarplot.png'
        if not os.path.exists(saveFolder / 'barPlots'):
            os.mkdir(saveFolder / 'barPlots')
        plt.savefig(saveFolder / 'barPlots' / fileName)
        plt.close()

        splot=sns.barplot(x='name',y='Spearman IC50',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("Spearman's rho ΔIC50", size=26)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        fileName = 'SpearmanIC50barplot.png'
        if not os.path.exists(saveFolder / 'barPlots'):
            os.mkdir(saveFolder / 'barPlots')
        plt.savefig(saveFolder / 'barPlots' / fileName)
        plt.close()

        splot=sns.barplot(x='name',y='Spearman Emax',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("Spearman's rho ΔEmax", size=26)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        fileName = 'SpearmanEmaxbarplot.png'
        if not os.path.exists(saveFolder / 'barPlots'):
            os.mkdir(saveFolder / 'barPlots')
        plt.savefig(saveFolder / 'barPlots' / fileName)
        plt.close()

        splot=sns.barplot(x='name',y='MSE IC50',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("MSE ΔIC50", size=26)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        fileName = 'MSEIC50barplot.png'
        if not os.path.exists(saveFolder / 'barPlots'):
            os.mkdir(saveFolder / 'barPlots')
        plt.savefig(saveFolder / 'barPlots' / fileName)
        plt.close()

        splot=sns.barplot(x='name',y='MSE Emax',data=df, palette=['#d684bd', '#3274a1', '#c03d3e', '#3a923a', '#e1812c', '#9372b2', '#7f7f7f', '#2e8e72', '#96ae81'])#, order=df['Model'])
        plt.xlabel("Model Used", size=26)
        plt.title('Performance for each model', size=34)
        plt.bar_label(splot.containers[0], size=26,label_type='center')
        plt.ylabel("MSE ΔEmax", size=26)

        #to save in fullscreen
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        fileName = 'MSEemaxbarplot.png'
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

    plot.axes.axline((0, 0), (0.1, 0.1), linewidth=4, ls="--", c=".3", color='r')
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

    plot.axes.axline((0, 0), (0.1, 0.1), linewidth=4, ls="--", c=".3", color='r')
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
