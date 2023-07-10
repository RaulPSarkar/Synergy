import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from sklearn.model_selection import KFold 


coeffs = Path(__file__).parent / 'datasets/CoefficientsLasso.csv'
coeffs = pd.read_csv(coeffs, index_col=0)

#drop all the intercepts (not needed, for now)
coeffs = coeffs[coeffs['features'] != '(Intercept)']
coeffs = coeffs.drop(['Drug_id'], axis=1)


coeffs = coeffs.drop_duplicates(subset=['Drug_name','features'], keep='last')
df = pd.pivot(coeffs, index='features', columns='Drug_name', values='EN_coef')
outputFile = Path(__file__).parent / "datasets/coefsProcessed.csv"

df.to_csv(outputFile)

#df.dropna(how='all', axis=0, inplace=True)
#drop all empty rows -> no rows are empty though