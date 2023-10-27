import pandas as pd
from pathlib import Path
import os

#i needed to make this cause i forgot to merge them all in the train file once

basePaths = [Path(__file__).parent / '../predictions' /'temp'/ 'rfrun115regularcrispr', Path(__file__).parent / '../predictions' /'temp'/ 'rfrun115regularge', Path(__file__).parent / '../predictions' /'temp'/ 'rfrun115regularproteomics']
numberOfFolds = 5

for basePath in basePaths:
    fullPredictions = []
    for fold in range(numberOfFolds):

        path = str(basePath)
        fullPath = path + str(fold) + '.csv'
        pred = pd.read_csv(fullPath)
        
        fullPredictions.append(pred)


    path = str(basePath)
    fullPath = path + '.csv'
    fullPredictions = pd.concat(fullPredictions, axis=0)
    fullPredictions.to_csv(fullPath, index=False)


