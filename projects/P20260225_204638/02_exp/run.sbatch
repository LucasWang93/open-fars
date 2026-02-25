#!/bin/bash
#SBATCH -J fars_P20260225_204638
#SBATCH -p scavenge_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH -o /nfs/roberts/project/pi_yz875/sw2572/fars_mvp/artifacts/slurm_logs/P20260225_204638/%j.out
#SBATCH -e /nfs/roberts/project/pi_yz875/sw2572/fars_mvp/artifacts/slurm_logs/P20260225_204638/%j.err

source /home/sw2572/Keys/env.sh
source $FARS_VENV/bin/activate
cd $FARS_ROOT

python3 src/compute/run_experiment_standalone.py /nfs/roberts/project/pi_yz875/sw2572/fars_mvp/projects/P20260225_204638 /nfs/roberts/project/pi_yz875/sw2572/fars_mvp
