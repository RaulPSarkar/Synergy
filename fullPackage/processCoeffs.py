import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 
import seaborn as sns
import matplotlib.pyplot as plt


#
minimumNonNull = 0 #threshold of minimum amount of non-null coefficients for gene to be maintained

coefficientValueThreshold = 0.02
minimumNumberAbove = 3 #the minimum number of coefficients that should be over the threshold above (i.e. 3 coefficients should be above 0.01 for the gene to be maintained)

#a second threshold, similar to the previous one
coefficientValueSecondThreshold = 0.08
minimumSecondNumberAbove = 1 #the minimum number of coefficients that should be over the threshold above (i.e. 3 coefficients should be above 0.01 for the gene to be maintained)


coeffs = Path(__file__).parent / 'datasets/CoefficientsLasso.csv'
coeffs = pd.read_csv(coeffs, index_col=0)

#drop all the intercepts (not needed, for now)
coeffs = coeffs[coeffs['features'] != '(Intercept)']
coeffs = coeffs.drop(['Drug_id'], axis=1)


coeffs = coeffs.drop_duplicates(subset=['Drug_name','features'], keep='last')
df = pd.pivot(coeffs, index='features', columns='Drug_name', values='EN_coef')
outputFile = Path(__file__).parent / "datasets/coefsProcessed.csv"


#####ADDING THRESHOLDS TO REDUCE TOTAL NUMBER OF GENES USED TO TRAIN THE MODEL
##############################################################################

df.fillna(0, inplace=True)
#TAKEN FROM https://stackoverflow.com/questions/26053849/counting-non-zero-values-in-each-column-of-a-dataframe-in-python
nonNullAmounts = df.astype(bool).sum(axis=1)
firstFiltered = df.loc[nonNullAmounts >= minimumNonNull] #just selecting rows above threshold (based on nonNullAmounts series)

aboveThresholdFrequency = firstFiltered[abs(firstFiltered) >= coefficientValueThreshold].count(axis=1) #needs to be the absolute value
secondFiltered = firstFiltered.loc[aboveThresholdFrequency >= minimumNumberAbove]


aboveSecondThresholdFrequency = secondFiltered[abs(secondFiltered) >= coefficientValueSecondThreshold].count(axis=1) #needs to be the absolute value
thirdFiltered = secondFiltered.loc[aboveSecondThresholdFrequency >= minimumSecondNumberAbove]


#sns.histplot(data=nonNullAmounts)
#plt.show()
#sns.histplot(data=aboveThresholdFrequency)
#plt.show()





##############################################################################


thirdFiltered.to_csv(outputFile)
