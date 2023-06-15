# Synergy

Before running the code, let's install some libraries:

pip install pubchempy
pip install rdkit
pip install cirpy
pip install chemspipy
pip install mahalanobis
pip install lightgbm
pip install keras_tuner
pip install tensorflow
pip install os
pip install pathlib
pip install pandas

More libraries than this are used, I didn't account for all of them.
Once your libraries are installed, we can start by processing our datasets in order to generate the dataframe that we'll be using to train our model.

Processing the dataset with processGDSC.py

Once you've selected which omics file to use to process the dataset with, you can run processGDSC.py.
Running processGDSC.py will concatenate pancreas_anchor_combo.csv, colon_anchor_combo.csv, and breast_anchor_combo.csv, and generates a dataframe called with an added SMILES A and SMILES B columns. 
This dataframe keeps only rows with known SMILES for both drugs (only 1 drug doesn't have a known SMILES). A cached version of drug names to SMILES has been generated, and is present at drug2smiles.txt for use during this process.
This dataframe also keeps only rows for which the cell line has omics data present.
Afterwards, it selects between duplicates (i.e. same drug pair-cell line triplets with different concentrations) using the Mahalanobis distance. The rows are shuffled before saving to the .csv, so that they don't have to be shuffled during training. This is to ensure that all model predictions have the experiments in the same order (for ensemble).
Finally, it saves the output to processedCRISPR.csv. This file contains all the experiments for which the cell line omics and drug SMILES are known. We will use its columns [Drug Smiles A, Drug Smiles B, Cell Line] as features, and the columns [Delta IC50, Delta Emax] as outcome variables.

Training the models with trainSKLearn.py

Now, we're ready to train our models using sklearn's functions. First, change modelName to the model you wish to train. If you wish to train with baseline, changing useBaselineInstead to True will override the selected model, and train the baseline model instead. The variable tunerTrials controls how many iterations are used for the hyperparameter tuning process. Currently, a list of 978 landmark genes from L1000 is being used to pre-select genes (all other gene columns are dropped) for improved performance and predictive accuracy.
10% of the data is used as validation data that is strictly used for the hyperparameter tuning process. For the remaining data, an 80-20 train-test split is applied, with 5-Fold cross validation (this split can be changed by changing the kFold variable). Predictions for delta Emax and delta IC50 are made for each test set on each fold, and then aggregated and saved onto the predictions/full folder.


Creating the scatter plot and bar graphs with makeGraphs.py

The makeGraphs.py file has 2 functions. First, it generates a file with statistics of every model in the predictionPaths/predictionNames variables (Pearson, R2, MSE, Spearman's rho). This file is saved onto modelStatsFolder.
Then, it generates and saves the graphs for every model in the predictionPaths/predictionNames variables onto the saveGraphsFolder directory.
To add a new model for which to run this for, just add the path of its predictions csv to predictionPaths, and the model name you wish to give it to predictionsName. These should both be added to the end of each array (or just to the same index).


Creating grouped scatter plots with groupedRegressionGraphs.py

