from SFC_Torch import SFcalculator
import reciprocalspaceship as rs
import torch
from torch.distributions.normal import Normal

pdb_file = '../SFcalculator_torch/tests/data/6ry3.pdb'
mtz_file = '../SFcalculator_torch/tests/data/6ry3_phases.mtz'

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
Fmodel_amplitude = torch.abs(Fmodel)

sfcalculator.Fo # structure factors from reducing data
sfcalculator.SigF # error of structure factors from reducing data

# Make a Gaussian distribution and calculate likelihood of Fmodel_amplitude

loss_per_sf = -Normal(sfcalculator.Fo,sfcalculator.SigF).log_prob(Fmodel_amplitude)
total_loss = torch.sum(loss_per_sf)

print(total_loss)

breakpoint()