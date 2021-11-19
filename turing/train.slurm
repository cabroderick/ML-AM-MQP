#!/bin/bash
#SBATCH --job-name=custom-mrcnn         # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4GB               # total memory per core (default is 4 GB per CPU-core)
#SBATCH --gres=gpu:4                    # number of gpus
#SBATCH --mail-type=all                 # email on job start, end and fault
#SBATCH --mail-user=cabroderick@wpi.edu # email to send to


source ~/project_env/bin/activate # full path to virtual env
python train.py custom_mrcnn_coco /home/cabroderick/mask_rcnn_coco.h5 # train model with coco weights loaded
deactivate
