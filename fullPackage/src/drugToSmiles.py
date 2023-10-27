import pubchempy
import cirpy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
#from chemspipy import ChemSpider
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np


def drugToSMILES(drugName, cached=False, cachedFile="drug2smiles.txt"):

    

    if(not cached):
        #cs = ChemSpider('') #requires key from chemspider
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
                    pass
                    #c2 = cs.search(drugName)


                    #if(c2):
                    #    return c2[0].smiles
                    #else:
                    #    return -1
    else:
        drug2smile = pd.read_csv(cachedFile, sep='\t')
        try:
            row = drug2smile.loc[drug2smile['name'] == drugName.strip()]
            return row.iloc[0,2]
        except:
            return -1
        

def SMILEStoFingerprint(smilesTable, smilesColumnName='SMILES_A'):
    #replace every row SMILES with ...
    fingerprints = []
    for index, row in smilesTable.iterrows():
        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(row[smilesColumnName]) , radius=2, bitInfo=bi, nBits=1024)
        fp = np.array(fp)
        #print(fp)
        fp = pd.DataFrame(data=[fp], index=[index] )
        #print(fp)
        fingerprints.append(fp)

    final = pd.concat(fingerprints)
    final["drug"] = smilesTable["drug"]
    final[smilesColumnName] = smilesTable[smilesColumnName]
    return final
