import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 


coeffs = Path(__file__).parent / 'datasets/CoefficientsLasso.csv'
coeffs = pd.read_csv(coeffs, index_col=0)

#drop all the intercepts (not needed?)
coeffs = coeffs[coeffs['features'] != '(Intercept)']
#drop all the GDSC2 (for now)
#drop id column
coeffs = coeffs.drop(['Drug_id'], axis=1)



print(coeffs)

coeffs = coeffs[coeffs['Dataset'] != 'GDSC2']
#print(coeffs)

coeffs = coeffs.drop_duplicates(subset=['Drug_name','features'])


#print(coeffs[coeffs.duplicated(subset=['Drug_name','features'], keep=False) == True].shape)


df = pd.pivot(coeffs, index='features', columns='Drug_name', values='EN_coef')

outputFile = Path(__file__).parent / "datasets/coefsProcessed.csv"

df.to_csv(outputFile)

print(df)