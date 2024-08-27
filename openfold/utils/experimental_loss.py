import os
import string
import numpy as np
import torch
import glob
from torch.distributions.normal import Normal
import reciprocalspaceship as rs
from SFC_Torch.Fmodel import SFcalculator
from openfold.np import protein, residue_constants
from openfold.utils.multi_chain_permutation import kabsch_rotation

chain_id_inverse_mapping = {n: cid for n, cid in enumerate(string.ascii_uppercase)}

train_data_dir_path = os.environ['TRAIN_DATA_DIR']
data_dir_path = os.environ['DATA_DIR']

residue_list = [residue_constants.restype_1to3[res] for res in residue_constants.restypes]
residue_list.append('OTHER')


def kabsch_align(output_atoms_positions, sfcalculator_corresponding_atoms):
    """
    Reference: https://hunterheidenreich.com/posts/kabsch_algorithm/
    Aligning the output_atoms_positions to sfcalculator_corresponding_atoms

    output_atoms_positions is P
    sfcalculator_corresponding_atoms is Q

    This code only computes the translation
    """

    # Find the centroid
    output_atoms_centroid = torch.mean(output_atoms_positions, axis=0)
    sfcalculator_atoms_centroid = torch.mean(sfcalculator_corresponding_atoms, axis=0)

    # Center the points
    output_atoms_positions_centered = output_atoms_positions - output_atoms_centroid
    sfcalculator_corresponding_atoms_centered = sfcalculator_corresponding_atoms - sfcalculator_atoms_centroid

    # Find the rotation
    rotation = kabsch_rotation(output_atoms_positions_centered, sfcalculator_corresponding_atoms_centered)

    # final points    
    output_atoms_positions_rotated = (output_atoms_positions_centered @ rotation.T) + sfcalculator_atoms_centroid

    rmsd_0 = torch.sqrt(torch.sum(torch.square(output_atoms_positions - sfcalculator_corresponding_atoms)))/output_atoms_positions.shape[1]
    rmsd_1 = torch.sqrt(torch.sum(torch.square(output_atoms_positions_rotated - sfcalculator_corresponding_atoms)))/output_atoms_positions_rotated.shape[1]
    print(f"RMSD was {rmsd_0} before alignment and is now {rmsd_1} after alignment.")

    return(output_atoms_positions_rotated)

def get_experimental_loss(outputs, batch):
    file_ids = []
    for j in range(batch['file_id'].shape[0]): # each protein of the batch
        file_ids.append(''.join([chr(i) for i in batch['file_id'][j]])) 

    mtz_files = []
    for file_id in file_ids:
        file_path = "../openfold_training/pdb_data/mtz_files/" + file_id + ".mtz"
        if os.path.exists(file_path):
            mtz_files.append(file_path)
        else:
            mtz_files.append(None)

    """ 
    output_proteins attributes: 'aatype', 'atom_mask', 'atom_positions', 'b_factors', 'chain_index', 'library', 'parents', 'parents_chain_index', 'remark', 'residue_index'
    """
    
    output_proteins = protein.from_prediction(batch, outputs, remove_leading_feature_dimension=False, library=torch)
    num_residues = output_proteins.atom_mask.shape[1]

    total_loss = 0
    for ind, file_id in enumerate(file_ids):
        if mtz_files[ind] is not None:

            pdb_file = train_data_dir_path + '/' + file_id + '.cif' # can be either .cif or .pdb
            mtz_file = data_dir_path + '/pdb_data/mtz_files/' + file_id + '.mtz' # ground truth download from the PDB for comparison
            sfcalculator = SFcalculator(pdb_file, mtz_file, expcolumns=['FP', 'SIGFP'], set_experiment=True, freeflag='FREE', testset_value=0, random_sample=True, device=batch['all_atom_positions'].device)

            # This is necessary before the following calculation, for the solvent percentage and grid size
            # Typically you only have to do it once
            sfcalculator.inspect_data(spacing=4.5, sample_rate=3.0, verbose=True, dynamic_spacing=True) # solvent percentage and grid size
            # The results will be stored in sfcalculator.Fprotein_HKL and sfcalculator.Fmask_HKL, used for future calculation
            # You can also return the tensor by Return=True

            """ 
            reorder the positions in openfold output to match those in sfcalculator.cra_name
            The positions in the reordered output should match those in sfcalculator._atom_pos_orth

            output_proteins.atom_mask == 1 for any existing atom
            for every unmasked atom in the output protein, we need: "chain_id"-"residue_index"-"amino_acid_code"-"atom_name"
            """
        
            """
            # This code snippet would get all the relevant quantities from sfcalculator:

            for atom in sfcalculator.cra_name:
                ref_chain_letter, ref_residue_index, ref_amino_acid_code, ref_atom_name = atom.split('-')
            """

            chain_id = np.tile(np.array([chain_id_inverse_mapping[i.cpu().numpy().item()] for i in output_proteins.chain_index[ind]])[:,None], [1,residue_constants.atom_type_num])
            residue_index = output_proteins.residue_index[ind][:,None].repeat(1,residue_constants.atom_type_num).cpu().numpy().astype('<U100')            
            amino_acid_code = np.tile(np.array([residue_list[i] for i in output_proteins.aatype[ind]])[:,None], [1,residue_constants.atom_type_num])
            atom_name = np.tile(np.array(residue_constants.atom_types)[None],[num_residues,1]) # ['N', 'CA', 'C', 'CB', 'O', 'CG', ... ]
            
            mask = output_proteins.atom_mask[ind].to(torch.float32).cpu().numpy() == 1
            output_cra_name = np.char.add(np.char.add(np.char.add(np.char.add(chain_id[mask], '-'), np.char.add(residue_index[mask], '-')), np.char.add(amino_acid_code[mask], '-')), atom_name[mask]).tolist()
            
            """
            Note that 'OTHER' amino acids and amino acids in the sequence but not in the PDB are ignored.
            """

            sfcalculator_inds = [sfcalculator.cra_name.index(item) for item in output_cra_name if item in sfcalculator.cra_name]
            used_output = [item in sfcalculator.cra_name for item in output_cra_name]

            """
            # Test
            aligned_pos = batch['all_atom_positions'][ind][output_proteins.atom_mask[ind]==1][used_output] # use to check
            print(aligned_pos)
            print(sfcalculator._atom_pos_orth[sfcalculator_inds])
            assert torch.all(aligned_pos==sfcalculator._atom_pos_orth[sfcalculator_inds])
            """

            output_atoms_positions = output_proteins.atom_positions[ind][output_proteins.atom_mask[ind]==1][used_output] # prediction
            sfcalculator_corresponding_atoms = sfcalculator.atom_pos_orth[sfcalculator_inds] # ground truth
            
            # Align the positions to some reference
            aligned_pos  = kabsch_align(output_atoms_positions, sfcalculator_corresponding_atoms)

            # Replace ground truth PDB with predicted positions of atoms
            sfcalculator._atom_pos_orth[sfcalculator_inds] = aligned_pos.type(torch.float32)

            sfcalculator.calc_fprotein(atoms_position_tensor=sfcalculator._atom_pos_orth, atoms_biso_tensor=None, atoms_occ_tensor=None, atoms_aniso_uw_tensor=None)
            sfcalculator.calc_fsolvent()

            # Get the unparameterized scales
            # stored in sfcalculator.kmasks, sfcalculator.uanisos, sfcalculator.kisos
            # If you want do a further optimization, there is 
            # sfcalculator.get_scales_adam() or sfcalculator.get_scales_lbfgs()
            # determines a set of scale values for each resolution bin
            sfcalculator.init_scales(requires_grad=True)
            # sfcalculator.get_scales_adam()

            # Get the Fmodel for future loss function construction
            Fmodel = sfcalculator.calc_ftotal()
            Fmodel_amplitude = torch.abs(Fmodel) # calculated from the output of openfold

            """ 
            sfcalculator.Fo # structure factors from reducing experimental data
            sfcalculator.SigF # error of structure factors from reducing experimental data
            """
            # Make a Gaussian distribution and calculate likelihood of Fmodel_amplitude

            loss_per_sf = -Normal(sfcalculator.Fo,sfcalculator.SigF).log_prob(Fmodel_amplitude)
            total_loss += torch.sum(loss_per_sf)

    print(total_loss)
    return(total_loss)
