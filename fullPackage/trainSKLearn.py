import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path



##########################
##########################
#Change
data = Path(__file__).parent / 'datasets/processedCRISPR.csv'
omics = Path(__file__).parent / 'datasets/crispr.csv.gz'
kFold = 5
##########################
##########################



data = pd.read_csv(data)
omics = pd.read_csv(omics, index_col=0)



def datasetToInput(data, omics):

    #print(omics.T)

    print("Generating Input Dataset. This may take a while...")
    test = data.merge(omics.T, left_on='CELLNAME', right_index=True)
    
    print(test)

    return data
    pass



datasetToInput(data,omics)

#cross validation



#loop hyperparameters/train



#leave one out validation