mamba create -n st python=3.10
mamba activate st
mamba install -y \
    -c conda-forge -c pytorch -c nvidia -c pyg \
    numpy matplotlib seaborn sympy pandas numba scikit-learn ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch torchvision torchaudio torchtriton pytorch-cuda=11.8 \
    pyg pytorch-scatter pytorch-sparse pytorch-cluster \
    wandb \
    ase python-lmdb h5py \
    cloudpickle \
    pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# install latest beta version of einops
pip install --upgrade --pre einops

# conda forge is out of date for torchmetrics/pytorch-lightning
pip install pytorch-lightning torchmetrics
# For the "lightning" package, we want to prevent pip from
# reinstalling an older version of pydantic.
# So we first install lightning, then pydantic.
pip install lightning
mamba install -y -c conda-forge pydantic

# for building pytorch cuda extensions
mamba install -c "nvidia/label/cuda-11.8.0" cuda

# for trajectory datasets
mamba install -c conda-forge pyarrow
