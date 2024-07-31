"""
batch is the input data
outputs is the output of OpenFold

> batch.keys()
> outputs.keys()

> outputs['final_atom_positions'] # batch_size x 256 (sequences cropped to 256 residues) x 37 x coordinates XYZ

atom37 representation: each heavy atom (i.e. does not include H atoms) corresponds to a given position in a 37-d array, e.g.
'C delta 1' has a specific spot
atoms not in the residue are zeroed out and indicated in a mask, see outputs['final_atom_mask'] which is batch_size x 256 x 37
In atom37, the representation for every kind of amino acid is the same, atom14 differs that there are only 14 slots, but each slot
refers to something different for different amino acids

SF Calculator creates a Gaussian of the distribution of possible structure factors
from experimental data. The loss is the likelihood of the output of OpenFold given the 
experimental distribution. The experimental data has a particular orientation which 
needs to be considered in computing the experimental loss. The OpenFold output should be rotated 
to best match the experimental output. Rotation details TBD 

SF Calculator needs the following values to compute experimental loss:
unit cell info, space group info, atom name info, atom position info, atoms B-factor info and atoms occupancy info

path to ground truth mtz:
cd ../openfold_training/pdb_data/mtz_files/*.mtz

protein.to_modelcif(output_protein) # for output_protein created with numpy
protein.to_pdb(output_protein) # for output_protein created with numpy

note: not all proteins have a corresponding mtz file (e.g. proteins with structure determined by cryo-EM)
"""
import torch
from openfold.np import protein, residue_constants
import os
import reciprocalspaceship as rs
import torch
from torch.distributions.normal import Normal
from SFC_Torch.Fmodel import SFcalculator

train_data_dir_path = os.environ['TRAIN_DATA_DIR']
data_dir_path = os.environ['DATA_DIR']


def format_openfold_output(output_proteins, ind, cra_name):
    # XXX write me
    # output_proteins[“ final_atom_positions”], output_proteins[“ atom_mask”]
    return atoms_position_tensor

def kabsch_align():
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

        output_proteins = protein.from_prediction(batch, outputs, library=torch)
        
        for ind, file_id in enumerate(file_ids):
            if mtz_files[ind] is not None:

                pdb_file = train_data_dir_path + '/' + file_id + '.cif' # can be either .cif or .pdb
                mtz_file = data_dir_path + '/pdb_data/mtz_files/' + file_id + '.mtz' # ground truth download from the PDB for comparison

                sfcalculator = SFcalculator(pdb_file, mtz_file, expcolumns=['FP', 'SIGFP'], set_experiment=True, freeflag='FREE', testset_value=0)

                breakpoint()

                # This is necessary before the following calculation, for the solvent percentage and grid size
                # Typically you only have to do it once
                sfcalculator.inspect_data(verbose=True) # solvent percentage and grid size

                # The results will be stored in sfcalculator.Fprotein_HKL and sfcalculator.Fmask_HKL, used for future calculation
                # You can also return the tensor by Return=True
                
                #reorder the positions based on topology
                sfcalculator.cra_name
                sfcalculator._atom_pos_orth

                # for every atom, we need: Chain-Residue_Index-3_letter_amino_acid_code-Atom_name

                output_proteins.atom_mask[ind] # go from this to Atom_name
                output_proteins.atom_positions[ind]
                output_proteins.b_factors[ind] # optional, plddt, not true b factor
                output_proteins.chain_index[ind] # go from this to Chain Letter

                # In data module, fix aatype, residue_index, chain_index

                batch['aatype'][ind] # go from this to 3 letter amino acid code # amino acid type, note that output_proteins.aatype is incorrectly implemented without the batch dimension
                output_proteins.residue_index[ind] # stays the same as Residue_Index # where the snip is taken from corresponding to the full protein

                ## XXX For multimer, may need parents and parents_chain_index

                # Chain
                import string
                chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}

                # Residue_Index
                # stays the same!
                output_proteins.residue_index[ind]

                # 3_letter_amino_acid code
                # XXX CHECK and CHANGE TO USE HHBLITS_AA_TO_ID if needed, check that this is what the DataModule uses
                output_proteins.aatype[ind]
                residue_constants.restypes_with_x # :=21, single letter, Reproduce it by taking 3-letter AA codes and sorting them alphabetically + unknown
                residue_constants.restype_1to3

                # Atom_name: atom_mask to Atom_name, residue_constants.residue_atoms
                residue_constants.atom_types # ['N', 'CA', 'C', 'CB', 'O', 'CG', ... ]

                # XXX unit test: use the values from the input (batch) and it should match the loss values when using the PDB

                # output_proteins attributes: 'aatype', 'atom_mask', 'atom_positions', 'b_factors', 'chain_index', 'library', 'parents', 'parents_chain_index', 'remark', 'residue_index'

                # batch.keys(): 'aatype', 'residue_index', 'seq_length', 'all_atom_positions', 'all_atom_mask', 'resolution', 'is_distillation', 'template_aatype', 'template_all_atom_mask', 'template_all_atom_positions', 'template_sum_probs', 'seq_mask', 'msa_mask', 'msa_row_mask', 'template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask', 'template_torsion_angles_sin_cos', 'template_alt_torsion_angles_sin_cos', 'template_torsion_angles_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', 'atom14_atom_is_ambiguous', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames', 'pseudo_beta', 'pseudo_beta_mask', 'backbone_rigid_tensor', 'backbone_rigid_mask', 'chi_angles_sin_cos', 'chi_mask', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat', 'use_clamped_fape', 'batch_idx', 'file_id', 'no_recycling_iters'
                atoms_position_tensor_pred_reordered = format_openfold_output(output_proteins, ind, sfcalculator.cra_name)
                
                # Align the positions to some reference
                aligned_pos  = kabsch_align(reorderd_pred, sfcalculator.atom_pos_orth)

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
