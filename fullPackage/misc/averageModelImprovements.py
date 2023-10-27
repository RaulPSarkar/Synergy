import pandas as pd
import statsmodels.api as sm 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy
from pathlib import Path
import os



resultPaths = [Path(__file__).parent / '../results' /'regularComparison'/ 'Pearson EmaxDifferences.csv', Path(__file__).parent / '../results' /'regularComparison'/ 'Pearson IC50Differences.csv', Path(__file__).parent / '../results' /'regularComparison'/ 'R2 EmaxDifferences.csv', Path(__file__).parent / '../results' /'regularComparison'/ 'R2 IC50Differences.csv']
resultNames = ['Pearson Emax', 'Pearson IC50', 'R2 Emax', 'R2 IC50']
outputFile = Path(__file__).parent / '../results' /'regularComparison'/ 'meanDifferences.csv'



counter = 0



allMeanResults = pd.DataFrame()

for j in resultPaths:

    results = pd.read_csv(j)
    resultName = resultNames[counter]
    counter += 1
    meanResults = results.mean()

    allMeanResults[resultName] = meanResults


print(allMeanResults)

allMeanResults.T.to_csv(outputFile)