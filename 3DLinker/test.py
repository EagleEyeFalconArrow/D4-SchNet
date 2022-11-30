import pytest
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import rdMolAlign
from joblib import Parallel, delayed
import frag_utils
import os, sys

data_set = "ZINC" # Options: ZINC, CASF
gen_smi_file = "generated_samples/generated_smiles.smi" # Path to generated molecules
train_set_path = "zinc/smi_train.txt" # Path to training set
n_cores = 1# Number of cores to use
verbose = True # Output results
restrict ="None"
pains_smarts_loc = "analysis/wehi_pains.csv" # Path to PAINS SMARTS
generated_smiles = frag_utils.read_triples_file(gen_smi_file)
in_mols = [smi[1] for smi in generated_smiles]
frag_mols = [smi[0] for smi in generated_smiles]
gen_mols = []
# drop invalid generations
for smi in generated_smiles:
    try:
        gen_mols.append(Chem.CanonSmiles(smi[2]))
    except:
        gen_mols.append("*")

# Remove dummy atoms from starting points
clean_frags = Parallel(n_jobs=n_cores)(delayed(frag_utils.remove_dummys)(smi) for smi in frag_mols)


def check_if_valid(gen_mol,clean_frag):
    if Chem.MolFromSmiles(gen_mol) == None:
        return False
        # gen_mols is chemically valid
    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(gen_mol), sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except:
        print('Chemical Invalid.')
        return False
    # gen_mols should contain both fragments
    if len(Chem.MolFromSmiles(gen_mol).GetSubstructMatch(Chem.MolFromSmiles(clean_frag))) == Chem.MolFromSmiles(
            clean_frag).GetNumAtoms():
        return True
    
    
def test_check_if_valid():
    for in_mol, frag_mol, gen_mol, clean_frag in zip(in_mols, frag_mols, gen_mols, clean_frags):
        assert(check_if_valid(gen_mol,clean_frag)==True)