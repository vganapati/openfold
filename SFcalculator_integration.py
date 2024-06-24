from SFC_Torch import SFcalculator

pdb_file = '../SFcalculator_torch/tests/data/1dur.pdb'
mtz_file = '../SFcalculator_torch/tests/data/1dur.mtz'

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
sfcalculator.init_scales(requires_grad=True)

# Get the Fmodel for future loss function construction
Fmodel = sfcalculator.calc_ftotal()

sfcalculator.summarize()