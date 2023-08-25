import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

#predictionPaths = [Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun70regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun71regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun72regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun73regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun74regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun75regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun76regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun77regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun78regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun79regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun710regularplusDrugs.csv', Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'lgbmrun711regularplusDrugs.csv']
#sensitivitySizeFractions = [0.01, 0.03, 0.06, 0.125, 0.17, 0.25, 0.375, 0.5, 0.625, 0.75, 0.85, 1] #sizes used for training (used to plot the model)
predictionBasePath = Path(__file__).parent / 'predictions' / 'final' / 'lgbm' / 'powerAnalysis'
sensitivitySizeFractions = [0.01, 0.03, 0.06, 0.1, 0.125, 0.15, 0.17, 0.25, 0.3, 0.375, 0.42, 0.5, 0.625, 0.75, 0.85, 0.9, 0.95, 0.98, 1] #trains the model with each of
totalIterations = 3 #number of iterations on which power analysis was done

resultsFolder =  Path(__file__).parent / 'results' / 'sens'
graphsFolder =  Path(__file__).parent / 'graphs' / 'sens'






#this is copied from makeGraphs, turn it into a function later
##############################################
####CREATE MODEL STATISTICS FILE
##############################################
counter = 0


it = 0
count = 0

fullStatsDF = [] 

for i in range(totalIterations):


    for j in range( len(sensitivitySizeFractions) ):


        file =  'lgbm' + str(count) + 'it' + str(it) + '.csv'
        fullFile = predictionBasePath / file
        print(fullFile)
        pred = pd.read_csv(fullFile)
        pred = pred.dropna(subset=['y_trueIC','y_predIC','y_trueEmax','y_predEmax'])
        #NAs dropped (they shouldn't exist) just in case the baseline has never seen the test drug pair
        count+=1
        modelName = sensitivitySizeFractions[counter]
        counter += 1
        if(count%len(sensitivitySizeFractions)==0):
            it+=1
            counter=0


        if not (count%len(sensitivitySizeFractions)==0 or count%len(sensitivitySizeFractions)==1 or count%len(sensitivitySizeFractions)==2):

            rhoEmax, p = spearmanr(pred['y_trueEmax'], pred['y_predEmax'])
            rhoIC50, p = spearmanr(pred['y_trueIC'], pred['y_predIC'])
            pearsonIC50, p = pearsonr(pred['y_trueIC'], pred['y_predIC'])
            pearsonEmax, p = pearsonr(pred['y_trueEmax'], pred['y_predEmax'])
            r2IC50 = r2_score(pred['y_trueIC'], pred['y_predIC'])
            r2Emax = r2_score(pred['y_trueEmax'], pred['y_predEmax'])
            mseIC50 = mean_squared_error(pred['y_trueIC'], pred['y_predIC'])
            mseEmax = mean_squared_error(pred['y_trueEmax'], pred['y_predEmax'])
            rho, p = spearmanr(pred[['y_trueIC', 'y_trueEmax']], pred[['y_predIC', 'y_predEmax']], axis=None)
            r2 = r2_score(pred[['y_trueIC', 'y_trueEmax']], pred[['y_predIC', 'y_predEmax']])

            ar = [modelName, pearsonIC50, rhoIC50, r2IC50, mseIC50, pearsonEmax, rhoEmax, r2Emax, mseEmax]
            df = pd.DataFrame(data=[ar], columns=['name', 'Pearson IC50', 'Spearman IC50', 'R2 IC50', 'MSE IC50', 'Pearson Emax',  'Spearman Emax', 'R2 Emax', 'MSE Emax'])
            fullStatsDF.append(df)
            
            
fullStatsDF = pd.concat(fullStatsDF, axis=0)



if not os.path.exists(resultsFolder):
    os.mkdir(resultsFolder)

if not os.path.exists(graphsFolder):
    os.mkdir(graphsFolder)

fullStatsDF.to_csv(resultsFolder / 'resultsNew.csv', index=False)
print(fullStatsDF)



for metricName in ['Pearson IC50', 'Spearman IC50', 'R2 IC50', 'Spearman Emax', 'R2 Emax', 'MSE Emax']:
    

    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)
    sns.set_style('whitegrid')
    plot=sns.lineplot(data=fullStatsDF, x="name", y="Pearson IC50", linewidth=5)
    
    figure = plt.gcf()
    figure.set_size_inches(32, 18)
    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)
    plt.xlabel("Fraction of Dataset used to Train/Test Model", size=36)
    plt.title('Performance of LGBM trained with Coefficient Data by Dataset Size', size=44)
    plt.ylabel(metricName, size=36)
    plot.ticklabel_format(style='plain')
    plot.set(xscale='log')
    plot.set(xticks=sensitivitySizeFractions)
    plot.set(xticklabels=sensitivitySizeFractions)
    plot.set(yticks=np.linspace(0, 1, 11))
    plt.setp(plot.get_xminorticklabels(), visible=False)

    fileName = 'sizeSensAnalysis' + metricName + '.png'
    plt.savefig(graphsFolder / fileName)
    plt.close()




#taken from https://keras.io/examples/keras_recipes/sample_size_estimate/
def fit_and_predict(train_acc, sample_sizes, pred_sample_size):
    
    """Fits a learning curve to model training accuracy results.

    Arguments:
        train_acc: List/Numpy Array, training accuracy for all model
                    training splits and iterations.
        sample_sizes: List/Numpy array, number of samples used for training at
                    each split.
        pred_sample_size: Int, sample size to predict model accuracy based on
                        fitted learning curve.
    """
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define mean squared error cost and exponential curve fit functions
    mse = keras.losses.MeanSquaredError()

    def exp_func(x, a, b):
        return a * x ** b

    # Define variables, learning rate and number of epochs for fitting with TF
    a = tf.Variable(0.0)
    b = tf.Variable(0.0)
    learning_rate = 0.01
    training_epochs = 5000

    # Fit the exponential function to the data
    for epoch in range(training_epochs):
        with tf.GradientTape() as tape:
            y_pred = exp_func(x, a, b)
            cost_function = mse(y_pred, mean_acc)
        # Get gradients and compute adjusted weights
        gradients = tape.gradient(cost_function, [a, b])
        a.assign_sub(gradients[0] * learning_rate)
        b.assign_sub(gradients[1] * learning_rate)
    print(f"Curve fit weights: a = {a.numpy()} and b = {b.numpy()}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc = exp_func(pred_sample_size, a, b).numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, exp_func(x_cont, a, b),  "r-", linewidth='6',label="Fitted exponential curve.")
    ax.set_ylabel("Pearson IC50.", fontsize=36)
    ax.set_xlabel("Training sample size.", fontsize=36)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_title("Pearson's r IC50 vs sample size.", fontsize=34)
    ax.legend(loc=(0.75, 0.75), fontsize=30)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    matplotlib.rc('font', size=35)
    matplotlib.rc('axes', titlesize=35)

    plt.tight_layout()
    plt.show()

    # The mean absolute error (MAE) is calculated for curve fit to see how well
    # it fits the data. The lower the error the better the fit.
    mse = keras.losses.MeanSquaredError()
    print(f"The mae for the curve fit is {mse(mean_acc, exp_func(x, a, b)).numpy()}.")


# We use the whole training set to predict the model accuracy
#fit_and_predict(fullStatsDF['Pearson IC50'].to_numpy(), sensitivitySizeFractions, pred_sample_size=2)