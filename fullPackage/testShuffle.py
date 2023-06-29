import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path

outputSMILEStoShuffledFingerprints = Path(__file__).parent / "datasets/smiles2shuffledfingerprints.csv"
outputSMILEStoFingerprints = Path(__file__).parent / "datasets/smiles2fingerprints.csv"
aaaa = pd.read_csv(outputSMILEStoShuffledFingerprints)
bbbb = pd.read_csv(outputSMILEStoFingerprints)
print(bbbb.sum(numeric_only=True).sort_values())
print(aaaa.sum(numeric_only=True).sort_values())

