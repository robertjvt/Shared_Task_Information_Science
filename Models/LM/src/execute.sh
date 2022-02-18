#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=d.a.tran@student.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB

module load cuDNN
module load Python/3.9.5-GCCcore-10.3.0
source /home/s2478935/Shared_Task_Information_Science/env/bin/activate
pip install -U pip wheel
pip install -r /home/s2478935/Shared_Task_Information_Science/requirements.txt

python3 train.py "$1"
