#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32GB       # total memory per core (default is 4 GB per CPU-core)
#SBATCH --gres=gpu:4             # number of gpus

source ./project_env/bin/activate # full path to virtual env
python job.py
deactivate
