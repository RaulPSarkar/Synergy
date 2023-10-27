import pandas as pd
import warnings
import sys
import numpy as np
from sklearn.utils import shuffle
from pathlib import Path
sys.path.append("..")
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib


##########################
##########################
ge = Path(__file__).parent / "../datasets/processedGeneExpression.csv"
saveFolder = Path(__file__).parent / "../graphs/mahalanobis/"
barplotXSize = 14 #x and y sizes to render the graphs on (smaller x makes bars thinner)
barplotYSize = 16
##########################
##########################


ge = pd.read_csv(ge)


extraText = '$ΔIC_{50}$'

for name in ['Delta Xmid', 'Delta Emax']:


    matplotlib.rc('font', size=28)
    matplotlib.rc('axes', titlesize=28)

    Ytext = 'Histogram of ' + extraText + ' in combination dataset'

    plt.ylabel('Number of samples', size=32, labelpad=16)
    plt.title(Ytext, size=38, pad = 18)
    plt.xlabel(extraText, size=42, labelpad=16)

    plot = sns.histplot(data=ge, x=name)


    figure = plt.gcf()
    figure.set_size_inches(barplotXSize, barplotYSize)

    fileName = 'mahalanobis' +name + '.png'
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    plt.savefig(saveFolder / fileName)
    plt.close()
    extraText = '$ΔE_{max}$'
