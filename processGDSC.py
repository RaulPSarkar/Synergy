
import pandas as pd
import pubchempy
import cirpy
from rdkit import Chem
from rdkit.Chem import AllChem
from chemspipy import ChemSpider
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


pancreas = pd.read_csv('pancreas_anchor_combo.csv')
colon = pd.read_csv('colon_anchor_combo.csv')
breast = pd.read_csv('breast_anchor_combo.csv')
yes = pd.concat([pancreas,colon, breast], axis=0)
#JOIN ALL DATASETS



drugNames = yes.groupby(['Library Name'])['Synergy?'].count()



def drugToSMILES(drugName, cached=False):

    if(not cached):
        cs = ChemSpider('ek03ZPZ3ITspWqMEWurgAQa4crlGhAJf')
        drugSmile =cirpy.resolve(drugName, 'smiles')
        try:
            
            m1 = Chem.MolFromSmiles(drugSmile)
            return drugSmile
            

        except TypeError:
            
            #try:
                
                
                results = pubchempy.get_compounds(drugName, 'name')
                

                
                if(results):
                    return results[0].isomeric_smiles
                else:
                    
                    c2 = cs.search(drugName)


                    if(c2):
                        return c2[0].smiles
                    else:
                        return -1
    else:
        drug2smile = pd.read_csv('drug2smiles.txt', sep='\t')
        try:
            row = drug2smile.loc[drug2smile['name'] == drugName.strip()]
            return row.iloc[0,2]
        except:
            return -1




print(drugNames)

smilesTable = pd.DataFrame(columns=['drug', 'SMILES_A'])

for name in range(drugNames.shape[0]):
    smiles = drugToSMILES( drugNames.index[name], True)
    if(smiles==-1):
        print(drugNames.index[name])
    else:
        smilesTable = smilesTable.append(pd.DataFrame([[drugNames.index[name], smiles]], columns=smilesTable.columns))

##CREATES A TABLE WITH ONLY THE DRUGS FOR WHICH SMILES ARE KNOWN (ALL BUT ONE IN THIS CASE)

out = yes.merge(smilesTable, left_on='Library Name', right_on='drug')
smilesTable.columns = ['drug', 'SMILES_B']

out2 = out.merge(smilesTable, left_on='Anchor Name', right_on='drug')







out2.rename(columns={'Library Name': 'NSC1', 'Anchor Name': 'NSC2', 'SDIM': 'CELLNAME'}, inplace=True)

out2['GROUP'] = out2['NSC1'] + '_' + out2['NSC2']
#out2.to_csv('dataset.csv')



crispr = pd.read_csv('transcriptomics.csv', index_col=0)
crispr = crispr.fillna(0)
crispr.columns.name = 'CELLNAME'
crisprT = crispr.T

#crisprT = pd.read_csv('crisprT.csv')
#crispr.T.to_csv('crisprT.csv')

crispr.T.to_csv('transcriptomicsT.csv')


print("here")


df = crisprT.merge(out2, on=['CELLNAME'])

df = df[['Tissue', 'CELLNAME', 'NSC1', 'NSC2', 'Anchor Conc', 'SMILES_A', 'SMILES_B', 'GROUP', 'Delta Xmid', 'Delta Emax']]

#df.rename(columns={'Synergy?': 'COMBOSCORE'}, inplace=True)


#dgi = pd.read_csv('dgi.csv')
#columnNames = []

#for column in dgi.columns:
#    if column in crisprT.columns:
#        columnNames.append(column)
#crisprT = crisprT[columnNames]

#print(crisprT)
#crisprT.to_csv('crisprT.csv')

df.drop_duplicates(inplace=True)

df.to_csv('datasetMultiExpressionBreast2.csv')

