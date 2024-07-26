import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from SFC_Torch.io import PDBParser

# export DATA_DIR=$CFS/m3562/users/vidyagan/openfold_training

DATA_DIR = os.environ['DATA_DIR']
path = DATA_DIR + '/pdb_data/mmcif_files/'
protein_filepaths = glob.glob(path + '*.cif')
num_truncate = 1000

residue_lengths = []

for protein_filepath in protein_filepaths[:num_truncate]: 
    protein_model = PDBParser(protein_filepath) 
    residue_length = protein_model.atom_pos.shape[0]
    residue_lengths.append(residue_length)
    print(residue_length)

print("Median Value ", np.median(residue_lengths))

plt.figure()
plt.hist(residue_lengths)
plt.title('Distribution of residue lengths in training dataset')
plt.savefig('residue_size_histogram.png')