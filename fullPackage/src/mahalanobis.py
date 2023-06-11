import pandas as pd
import numpy as np
import scipy as sp
from scipy import linalg
from mahalanobis import Mahalanobis


def mahalanobisFunc(data, columns, dropSubset, number=100000):

    dataMaha = data[columns]
    mah1D = Mahalanobis(dataMaha.to_numpy(), number)
    distances = mah1D.calc_distances(dataMaha.to_numpy())
    data['mahalanobis'] = distances
    dataFinal = data.sort_values("mahalanobis", ascending = False).drop_duplicates(subset=dropSubset, keep="first")
    dataFinal = dataFinal.sample(frac=1).reset_index(drop=True)
    return dataFinal