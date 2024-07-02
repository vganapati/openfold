from SFC_Torch import SFcalculator
import reciprocalspaceship as rs
import numpy as np
import torch

# pdb_file = '../SFcalculator_torch/tests/data/1dur.pdb'
# mtz_file = '../SFcalculator_torch/tests/data/1dur.mtz'

pdb_file = '../SFcalculator_torch/tests/data/6ry3.pdb'
mtz_file = '../SFcalculator_torch/tests/data/6ry3_phases.mtz'

sfcalculator = SFcalculator(pdb_file, mtz_file, expcolumns=['FP', 'SIGFP'], set_experiment=True, freeflag='FREE', testset_value=0)

print("N_HKL: ", len(sfcalculator.HKL_array))
print("N_atoms: ", sfcalculator.n_atoms)

# This is necessary before the following calculation, for the solvent percentage and grid size
# Typically you only have to do it once
sfcalculator.inspect_data(verbose=True)

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
sfcalculator.summarize()

"""
rs.read_mtz(mtz_file).FP # equivalent to sfcalculator.Fo (structure factors from reducing data)
rs.read_mtz(mtz_file).SIGFP # equivalent to sfcalculator.SigF (error of structure factors from reducing data)
rs.read_mtz(mtz_file).FC # deposited calculated value (structure factors that correspond to final solved atomic coordinates)
rs.read_mtz(mtz_file).get_hkls() # equivalent to sfcalculator.HKL_array
rs.read_mtz(mtz_file)["FC"][0,0,4]
"""

# check why # of HKLs is different

hkls_sfcalc = sfcalculator.HKL_array
hkls_mtz = rs.read_mtz(mtz_file).get_hkls()
max_hkls = hkls_mtz.shape[0]

for ind in range(max_hkls):
    hkls_mtz_i = hkls_mtz[ind]
    if np.any(np.all((hkls_sfcalc-hkls_mtz_i)==0, axis=1)):
        pass
    else:
        print(hkls_mtz_i)

"""
HKLs in the mtz file, but not in sfcalculator
[ 1  0 52]
[2 1 6]
[ 4  1 58]
[ 5  4 45]
[ 7  0 42]
[15 12 40]
[17  9 32]
"""

breakpoint()