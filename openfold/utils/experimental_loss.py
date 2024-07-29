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
from openfold.np import protein
import os
import reciprocalspaceship as rs
import torch
from torch.distributions.normal import Normal
from SFC_Torch.Fmodel import SFcalculator

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

        
        # XXX watch out if batch_idx is used and file_id is hardcoded!
        output_proteins = protein.from_prediction(batch, outputs, library=torch)


        breakpoint()
        # SF_calculator

        train_data_dir_path = os.environ['TRAIN_DATA_DIR']

        pdb_file = train_data_dir_path + '/1jux.cif' # can be either .cif or .pdb
        mtz_file = train_data_dir_path + '/1jux.mtz' # ground truth download from the PDB for comparison

        sfcalculator = SFcalculator(pdb_file, mtz_file, expcolumns=['FP', 'SIGFP'], set_experiment=True, freeflag='FREE', testset_value=0)

        # This is necessary before the following calculation, for the solvent percentage and grid size
        # Typically you only have to do it once
        sfcalculator.inspect_data(verbose=True) # solvent percentage and grid size

        # The results will be stored in sfcalculator.Fprotein_HKL and sfcalculator.Fmask_HKL, used for future calculation
        # You can also return the tensor by Return=True
        sfcalculator.calc_fprotein(atoms_position_tensor=None, atoms_biso_tensor=None, atoms_occ_tensor=None, atoms_aniso_uw_tensor=None)
        sfcalculator.calc_fsolvent()

        # Get the unparameterized scales
        # stored in sfcalculator.kmasks, sfcalculator.uanisos, sfcalculator.kisos
        # If you want do a further optimization, there is 
        # sfcalculator.get_scales_adam() or sfcalculator.get_scales_lbfgs()
        # determines a set of scale values for each resolution bin
        sfcalculator.init_scales(requires_grad=True)

        # Get the Fmodel for future loss function construction
        Fmodel = sfcalculator.calc_ftotal()
        Fmodel_amplitude = torch.abs(Fmodel) # calculated from the output of openfold

        sfcalculator.Fo # structure factors from reducing experimental data
        sfcalculator.SigF # error of structure factors from reducing experimental data

        # Make a Gaussian distribution and calculate likelihood of Fmodel_amplitude

        loss_per_sf = -Normal(sfcalculator.Fo,sfcalculator.SigF).log_prob(Fmodel_amplitude)
        total_loss = torch.sum(loss_per_sf)

        print(total_loss)
