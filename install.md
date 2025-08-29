# Installation

## Install the UV Package Manager
In this workshop, we use the uv package manager.
Please install [UV](https://docs.astral.sh/uv/) first.

You need [Git](https://github.com/git-guides/install-git) to clone (to your computer) the repository.

We recommend to use [VSCode](https://code.visualstudio.com/download) as a coding environment.

## Install the environment
We create virtual environment in the repository using uv:

```bash
uv venv .venv --python=3.11
source .venv/bin/activate

uv pip install torch==2.3.1 torchvision 

uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

uv pip install -r requirements.txt
```
