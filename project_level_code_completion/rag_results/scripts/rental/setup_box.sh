#!/usr/bin/env bash
# One-time setup of a rented Ubuntu box (NVIDIA driver present, >= 32 GB GPU memory).
# The repository is private, so no git credentials go on the box: build a tarball of the committed code on your PC
#   (from the repository root, after committing):  git archive --format=tar.gz -o lca_project.tar.gz HEAD project_level_code_completion
# copy it (scp -P <port> lca_project.tar.gz root@<host>:~/ ; also copy this rental/ folder, or take it from the tarball), then on the box:
#   bash ~/lca-repo/project_level_code_completion/rag_results/scripts/rental/setup_box.sh ~/lca_project.tar.gz
# Afterwards log in to Hugging Face with a READ-ONLY token (revoke it when the runs are done): `huggingface-cli login`.
set -e
TARBALL=${1:?usage: setup_box.sh <path-to-lca_project.tar.gz>}
SUDO=""; command -v sudo > /dev/null && SUDO=sudo
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y software-properties-common curl git build-essential openssh-client
$SUDO add-apt-repository -y ppa:deadsnakes/ppa          # Ubuntu 22.04 has no python3.11 otherwise
$SUDO apt-get update -y
$SUDO apt-get install -y python3.11 python3.11-venv python3.11-dev
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
python3.11 -m pip install --user poetry
export PATH=$HOME/.local/bin:$PATH
mkdir -p $HOME/lca-repo && tar -xzf "$TARBALL" -C $HOME/lca-repo       # gives $HOME/lca-repo/project_level_code_completion
cd $HOME/lca-repo/project_level_code_completion
poetry env use python3.11
poetry install                                   # torch 2.14.0 etc. are pinned in pyproject.toml / poetry.lock
mkdir -p $HOME/ts_test && cd $HOME/ts_test
git clone --branch v0.21.0 https://github.com/tree-sitter/tree-sitter-python   # tag v0.21.0: ABI matches tree-sitter 0.20.4
echo "--- environment check (the local machine has torch 2.14.0+cu130, transformers 5.17.0):"
cd $HOME/lca-repo/project_level_code_completion
poetry run python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
echo "Next: huggingface-cli login (read-only token), then: setsid nohup bash run_A.sh (or run_B.sh / run_C.sh) > /dev/null 2>&1 &"
