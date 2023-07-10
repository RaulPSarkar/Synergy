import numpy as np
import sys
import pandas as pd
import tensorflow as tf
import yaml
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
import keras_tuner
sys.path.append("..")
from src.buildDLModel import buildDL
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import os


coeffs = Path(__file__).parent / 'datasets/coefsProcessed.csv'
coeffs = pd.read_csv(coeffs)

sparr = coeffs.apply(pd.arrays.SparseArray)
print (sparr.sparse.density)

