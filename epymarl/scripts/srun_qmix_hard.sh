#!/bin/sh
#SBATCH --time=100:00:00          # Maximum run time in hh:mm:ss
#SBATCH --mem=64000             # Maximum memory required (in megabytes)
#SBATCH --job-name=lbh-hard-qmix        # Job name (to track progress)
#SBATCH --partition=dappt # Partition on which to run job
#SBATCH --gres=gpu:1        # Don't change this, it requests a GPU
#SBATCH --constraint=gpu_32gb  # will request a GPU with 16GB of RAM, independent of the type of card
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --error=/work/dappt/zonglin/level-based-hacking/epymarl/scripts/logs/job.%J.err
#SBATCH --output=/work/dappt/zonglin/level-based-hacking/epymarl/scripts/logs/job.%J.out

module purge
module load cuda
module load anaconda
module load git

conda activate $WORK/assets/conda-envs/marl

srun python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=100 env_args.key="Hacking-4p-Hard-v0" common_reward=True save_model=True save_model_interval=500000
                        