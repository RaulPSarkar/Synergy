#THIS PART IS ESPECIALLY IMPORTANT FOR PROTEOMICS, DUE TO ALL THE MISSING VALUES
#(need to normalize per GENE to center 0 - because of mv's that are imputated to 0)
import pandas as pd
import sys
import numpy as np
sys.path.append("..")
from pathlib import Path
from scipy.stats import zscore



##########################
##########################
crisprOmics = Path(__file__).parent / 'datasets/crispr.csv.gz'
transcriptomics = Path(__file__).parent / 'datasets/transcriptomics.csv.gz'
proteomics = Path(__file__).parent / 'datasets/proteomics.csv.gz'

crisprOutput = Path(__file__).parent / 'datasets/crisprProcessed.csv.gz'
transcriptomicOutput = Path(__file__).parent / 'datasets/transcriptomicsProcessed.csv.gz'
proteomicOutput = Path(__file__).parent / 'datasets/proteomicsProcessed.csv.gz'
##########################
##########################



crisprOmics = pd.read_csv(crisprOmics, index_col=0)
transcriptomics = pd.read_csv(transcriptomics, index_col=0)
proteomics = pd.read_csv(proteomics, index_col=0)

#wowow this is great code
for omics, output in zip( [crisprOmics, transcriptomics, proteomics], [crisprOutput, transcriptomicOutput, proteomicOutput] ):
    #normalize per GENE, (and then imputate to 0 for proteomics)
    omics = omics.T #its easier to operate with columns + im lazy

    counter = 0
    numGenes = len(omics.columns)
    for column in omics.columns:
        omics[column] = zscore(omics[column], nan_policy='omit')
        if(counter%50==0):
            print(str(counter) + '/' + str(numGenes)) #just to have an idea of how long its gonna take
        counter+=1
        
    omics = omics.T #back to original format
    omics = omics.fillna(0)
    omics = omics.astype('float16')
    omics.to_csv(output)

