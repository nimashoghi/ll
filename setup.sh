mamba create -n fm python=3.11
mamba activate fm

mamba install -y \
    -c pytorch -c nvidia -c pyg \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    pyg pytorch-scatter pytorch-sparse pytorch-cluster

mamba install -y \
    -c conda-forge \
    numpy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle \
    "pydantic>2" \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval
