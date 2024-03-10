mamba create -n fm python=3.11
mamba activate fm

# mamba install -y \
#     -c pytorch -c nvidia -c pyg \
#     pytorch torchvision torchaudio pytorch-cuda=12.1 \
#     pyg pytorch-scatter pytorch-sparse pytorch-cluster

# PyG is not really necessary for `ll`
mamba install -y \
    -c pytorch -c nvidia \
    pytorch torchvision torchaudio pytorch-cuda=12.1

mamba install -y \
    -c conda-forge \
    numpy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle \
    "pydantic>2" \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Install jaxtyping
pip install jaxtyping
