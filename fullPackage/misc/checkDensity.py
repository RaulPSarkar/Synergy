import numpy as np
import sys
import pandas as pd
import tensorflow as tf
import yaml
from pathlib import Path
import keras_tuner
sys.path.append("..")
import os


coeffs = Path(__file__).parent / '../datasets/coefsProcessed.csv'
coeffs = pd.read_csv(coeffs)

sparr = coeffs.apply(pd.arrays.SparseArray)
print (sparr.sparse.density)

