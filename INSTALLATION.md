# Setup on NERSC Perlmutter

Using CUDA 12 directions available [here](https://openfold.readthedocs.io/en/latest/Installation.html).

Start installation:
```
export USERNAME=your_nersc_username
cd $CFS/m3562/users/$USERNAME
git clone https://github.com/vganapati/openfold.git
cd openfold
git checkout pl_upgrades

module load conda
mamba env create -p /global/cfs/cdirs/m3562/users/$USERNAME/openfold_env -f environment.yml

conda activate /global/cfs/cdirs/m3562/users/$USERNAME/openfold_env
```

Check install thus far:
```
nvcc --version
```
Should return CUDA 12.2. This is a slight mismatch with the CUDA used to compile PyTorch, but should not pose a problem in the install.

Check PyTorch has GPU support:
```
python # enter Python command line
import torch
torch.cuda.is_available() # Should be True
torch.cuda.device_count()
torch.cuda.current_device()
quit()
```

Finish installation:
```
./scripts/install_third_party_dependencies.sh

conda deactivate
conda activate /global/cfs/cdirs/m3562/users/$USERNAME/openfold_env

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

bash scripts/download_alphafold_params.sh openfold/resources/
bash scripts/download_openfold_params.sh openfold/resources/
bash scripts/download_openfold_soloseq_params.sh openfold/resources/
```

Execute unit tests in an interactive session:
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3562

./scripts/run_unit_tests.sh
```

After install, need to do the following in `$HOME` to clean up tarballs: 
```
cd $HOME
conda clean -a
```

# Startup after setup

For the next log-in, follow these steps to run unit tests:
```
module load conda
export USERNAME=your_username
cd $CFS/m3562/users/$USERNAME/openfold

conda activate /global/cfs/cdirs/m3562/users/$USERNAME/openfold_env

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3562

./scripts/run_unit_tests.sh
```

# Known issues

This test is currently failing:
```
FAIL: test_compare_model (tests.test_deepspeed_evo_attention.TestDeepSpeedKernel)
Run full model with and without using DeepSpeed Evoformer attention kernel
```

# Removing mamba environment

If necessary:
```
module load conda
mamba remove -p /global/cfs/cdirs/m3562/users/$USERNAME/openfold_env --all
```
