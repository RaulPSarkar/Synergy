# Synergy

Before running the code, let's install the required dependencies. You can install all required dependencies by running:

```
pip install -r requirements.txt
```

Once your libraries are installed, we can start by processing our datasets in order to generate the dataframe that we'll be using to train our model.


## Processing the dataset with processGDSC.py

Once you've selected which omics file to use to process the dataset with, you can run processGDSC.py.
Running processGDSC.py will concatenate pancreas_anchor_combo.csv, colon_anchor_combo.csv, and breast_anchor_combo.csv, and generates a dataframe called with an added SMILES A and SMILES B columns. 
This dataframe keeps only rows with known SMILES for both drugs (only 1 drug doesn't have a known SMILES). A cached version of drug names to SMILES has been generated, and is present at drug2smiles.txt for use during this process. It also generates a smiles2fingerprints.csv files, with the fingerprint of each SMILES, and if setting shuffleFingerprintBits to True, additionally generates a smiles2shuffledfingerprints.csv, where the bits of each row are randomly shuffled.

This dataframe also keeps only rows for which the cell line has omics data present.
Afterwards, it selects between duplicates (i.e. same drug pair-cell line triplets with different concentrations) using the Mahalanobis distance. The rows are shuffled before saving to the .csv (although this step is irrelevant since they're reshuffled during training, I might remove later).
Finally, it saves the output to processedCRISPR.csv. This file contains all the experiments for which the cell line omics and drug SMILES are known. We will use its columns [Drug Smiles A, Drug Smiles B, Cell Line] as features, and the columns [Delta IC50, Delta Emax] as outcome variables.

## Training the models with trainSKLearn.py

Now, we're ready to train our models using sklearn's functions. First, change modelName to the model you wish to train. If you wish to train with baseline, changing useBaselineInstead to True will override the selected model, and train the baseline model instead. The variable tunerTrials controls how many iterations are used for the hyperparameter tuning process. Currently, a list of 978 landmark genes from L1000 is being used to pre-select genes (all other gene columns are dropped) for improved performance and predictive accuracy.
10% of the data is used as validation data that is strictly used for the hyperparameter tuning process. For the remaining data, an 80-20 train-test split is applied, with 5-Fold cross validation (this split can be changed by changing the kFold variable). Predictions for delta Emax and delta IC50 are made for each test set on each fold, and then aggregated and saved onto the predictions/full folder.

## Training the models with trainDL.py

This code is awfully similar to trainSKLearn, so I should merge the two. It uses Keras Hyperband.


## Creating the scatter plot and bar graphs with makeGraphs.py

The makeGraphs.py file has 2 functions. First, it generates a file with statistics of every model in the predictionPaths/predictionNames variables (Pearson, R2, MSE, Spearman's rho). This file is saved onto modelStatsFolder.
Then, it generates and saves the graphs for every model in the predictionPaths/predictionNames variables onto the saveGraphsFolder directory.
To add a new model for which to run this for, just add the path of its predictions csv to predictionPaths, and the model name you wish to give it to predictionsName. These should both be added to the end of each array (or just to the same index).


## Creating grouped scatter plots with groupedRegressionGraphs.py

The groupedRegressionGraphs.py loops over the predictionPaths/predictionNames files. For each file, it groups the data by the library drug, and performs an OLS regression on each of those groups. The R2 of each regression is then sorted from highest to lowest, in order to determine for which library drugs the model makes its best/worst predictions. Scatter plots for the top x best/worst library drugs for each model are stored in graphsFolder, alongside a boxplot of the R2 distribution for the library drugs.
Ideally, the grouping would be done by Library and Anchor drugs, but unfortunately, such groupings would result in very little data to regress with per group.


## Todo

Auto-detect number of genes
Auto-detect number of fingerprint bits (to make it easier to add one-hot encoding)
...