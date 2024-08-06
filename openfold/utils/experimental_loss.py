import numpy as np
import torch
from openfold.np import protein, residue_constants
import os
import reciprocalspaceship as rs
import torch
from torch.distributions.normal import Normal
from SFC_Torch.Fmodel import SFcalculator
import string

chain_id_inverse_mapping = {n: cid for n, cid in enumerate(string.ascii_uppercase)}

train_data_dir_path = os.environ['TRAIN_DATA_DIR']
data_dir_path = os.environ['DATA_DIR']

residue_list = [residue_constants.restype_1to3[res] for res in residue_constants.restypes]
residue_list.append('OTHER')

def format_openfold_output(output_proteins, ind, cra_name):
    # XXX write me
    # output_proteins[“ final_atom_positions”], output_proteins[“ atom_mask”]
    return atoms_position_tensor

def kabsch_align():
     # XXX write me
     pass

def fape_align():
     # XXX write me
     pass



def get_experimental_loss(outputs, batch):
        file_ids = []
        for j in range(batch['file_id'].shape[0]): # each protein of the batch
            file_ids.append(''.join([chr(i) for i in batch['file_id'][j]])) 

        mtz_files = []
        for file_id in file_ids:
            try:
                mtz_files.append("../openfold_training/pdb_data/mtz_files/" + file_id + ".mtz")
            except FileNotFoundError:
                mtz_files.append(None)

        """ 
        output_proteins attributes: 'aatype', 'atom_mask', 'atom_positions', 'b_factors', 'chain_index', 'library', 'parents', 'parents_chain_index', 'remark', 'residue_index'
        """
        
        output_proteins = protein.from_prediction(batch, outputs, remove_leading_feature_dimension=False, library=torch)
        num_residues = output_proteins.atom_mask.shape[1]

        for ind, file_id in enumerate(file_ids):
            if mtz_files[ind] is not None:

                pdb_file = train_data_dir_path + '/' + file_id + '.cif' # can be either .cif or .pdb
                mtz_file = data_dir_path + '/pdb_data/mtz_files/' + file_id + '.mtz' # ground truth download from the PDB for comparison

                sfcalculator = SFcalculator(pdb_file, mtz_file, expcolumns=['FP', 'SIGFP'], set_experiment=True, freeflag='FREE', testset_value=0)

                # This is necessary before the following calculation, for the solvent percentage and grid size
                # Typically you only have to do it once
                sfcalculator.inspect_data(verbose=True) # solvent percentage and grid size

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
                
                mask = output_proteins.atom_mask[ind].cpu().numpy() == 1
                output_cra_name = np.char.add(np.char.add(np.char.add(np.char.add(chain_id[mask], '-'), np.char.add(residue_index[mask], '-')), np.char.add(amino_acid_code[mask], '-')), atom_name[mask]).tolist()
                
                for item in output_cra_name:
                    print(item)
                    try:
                        sfcalculator.cra_name.index(item)
                    except ValueError:
                         breakpoint()
                
                breakpoint()
                sfcalculator_inds = [sfcalculator.cra_name.index(item) for item in output_cra_name if item in sfcalculator.cra_name]

                breakpoint()
                output_proteins.atom_positions[ind][output_proteins.atom_mask[ind]==1]

                atoms_position_tensor_pred_reordered = format_openfold_output(output_proteins, ind, sfcalculator.cra_name)
                
                # Align the positions to some reference
                aligned_pos  = kabsch_align(reorderd_pred, sfcalculator.atom_pos_orth)
                fape_align

                # XXX Need to align and replace

                # XXX use plddt for atoms_biso_tensor?
                sfcalculator.calc_fprotein(atoms_position_tensor=aligned_pos+atoms_not_solved, atoms_biso_tensor=None, atoms_occ_tensor=None, atoms_aniso_uw_tensor=None)
                sfcalculator.calc_fsolvent()

                # Get the unparameterized scales
                # stored in sfcalculator.kmasks, sfcalculator.uanisos, sfcalculator.kisos
                # If you want do a further optimization, there is 
                # sfcalculator.get_scales_adam() or sfcalculator.get_scales_lbfgs()
                # determines a set of scale values for each resolution bin
                sfcalculator.get_scales_adam(requires_grad=True)

                # Get the Fmodel for future loss function construction
                Fmodel = sfcalculator.calc_ftotal()
                Fmodel_amplitude = torch.abs(Fmodel) # calculated from the output of openfold

                """ 
                sfcalculator.Fo # structure factors from reducing experimental data
                sfcalculator.SigF # error of structure factors from reducing experimental data
                """
                
                # Make a Gaussian distribution and calculate likelihood of Fmodel_amplitude

                loss_per_sf = -Normal(sfcalculator.Fo,sfcalculator.SigF).log_prob(Fmodel_amplitude)
                total_loss = torch.sum(loss_per_sf)

                print(total_loss)
