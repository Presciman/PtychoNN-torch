#Quick Start Guide of PtychoNN on ThetaGPU

#STEP 1: Download PtychoNN code and data
git clone https://github.com/mcherukara/PtychoNN.git

#STEP 2: Use conda on ThetaGPU
source /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/setup.sh

#STEP 3: Copy base environment as your own environment (e.g., dist)
conda create --name dist --clone base

#STEP 4: Install tqdm, sklearn, scikit-image, mpi4py
pip3 install tqdm
conda install -c anaconda scikit-learn
conda install -c anaconda scikit-image
conda install -c conda-forge mpi4py

#STEP 5: Put train_mpi.py under PtychoNN folder
#STEP 6: Modify Line 192 and 193 in train_mpi.py to the PtychoNN/data/20191008_39_diff.npz and 20191008_39_amp_pha_10nm_full.npy, respectively.

#Case 1: Run with mpi on Single node using 8 GPUs
mpirun -np 8 python3 -u train_mpi.py --epochs 100 --model_save_path /path/to/save/model/ckpt

#Case 2:  Run with mpi on 2 node using 16 GPUs (8 GPUs each node)
mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np 16 -npernode 8 --hostfile ${COBALT_NODEFILE} python3 -u train_mpi.py --epochs 100 --model_save_path /path/to/save/model/ckpt

