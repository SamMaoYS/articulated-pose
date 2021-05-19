
CONDA_ENV_DIR=/home/sam/anaconda3/envs/ansch

export PATH="/usr/local/cuda/bin/:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow:$LD_LIBRARY_PATH"
