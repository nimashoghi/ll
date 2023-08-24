mamba create -n fm python=3.10
mamba activate fm

mamba install -y \
    -c pytorch -c nvidia -c pyg \
    pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 \
    pyg pytorch-scatter pytorch-sparse pytorch-cluster

mamba install -y \
    -c conda-forge \
    numpy matplotlib seaborn sympy pandas numba scikit-learn ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    wandb \
    ase python-lmdb h5py \
    cloudpickle \
    pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# install latest beta version of einops
pip install --upgrade --pre einops
