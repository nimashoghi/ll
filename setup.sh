mamba create -n fm python=3.11
mamba activate fm
mamba install -y \
    -c conda-forge -c pytorch -c nvidia -c pyg \
    numpy matplotlib seaborn sympy pandas numba scikit-learn ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    pyg pytorch-scatter pytorch-cluster \
    pytorch-lightning torchmetrics \
    wandb \
    ase python-lmdb h5py \
    cloudpickle \
    pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# install latest beta version of einops
pip install --upgrade --pre einops
