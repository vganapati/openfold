"""
This script downloads the corresponding MTZ file for every mmCIF file in the training dataset.

Usage:
from the openfold directory:
python3 download_mtz.py 
"""

import os
import requests
import glob
from numpy import savetxt

def download_mtz_files(file_list, mmcif_directory, output_dir):
    base_url = "https://edmaps.rcsb.org/coefficients/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ids_sf = []
    ids_no_sf = []
    for pdb_id in file_list:
        url = f"{base_url}{pdb_id}.mtz"
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = os.path.join(output_dir, f"{pdb_id}.mtz")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded MTZ file for: {pdb_id}")
            ids_sf.append(pdb_id)
        else:
            print(f"Failed to download MTZ file for: {pdb_id}")
            ids_no_sf.append(pdb_id)
    savetxt(os.path.join(output_dir,'ids_structure_factors.dat'), ids_sf, fmt='%s')
    savetxt(os.path.join(output_dir,'ids_no_structure_factors.dat'), ids_no_sf, fmt='%s')


if __name__ == "__main__":
    mmcif_directory = "../openfold_training/pdb_data/mmcif_files"
    full_mmcif_file_paths = glob.glob(mmcif_directory + "/*.cif")

    # List of PDB IDs
    pdb_ids = [os.path.splitext(os.path.basename(path))[0] for path in full_mmcif_file_paths]

    # Output directory to save the .mtz files
    output_directory = "../openfold_training/pdb_data/mtz_files"

    # Download the .mtz files
    download_mtz_files(pdb_ids, mmcif_directory, output_directory)
