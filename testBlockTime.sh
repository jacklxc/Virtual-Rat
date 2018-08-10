#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Number of nodes
#SBATCH -t 7-00:00 # Runtime in D-HH:MM
##SBATCH -p centos7 # Partition to submit to
#SBATCH --mem=8000 # Memory pool for all cores, MB
##SBATCH -o TestBlockTime.o # File to which STDOUT will be written 
#SBATCH -e TestBlockTime.e # File to which STDERR will be written 
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xl1066@nyu.edu # Email or Chinese mobile phone NO. to which notifications will be sent
##SBATCH --constraint=2650v4 # the Features of the nodes
#SBATCH --job-name v-rat
#SBATCH --output TestBlockTime-log-%J-%a.txt
#SBATCH --array=0-69
## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## start an ipcluster instance and launch jupyter server
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST 
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /etc/profile.d/modules.sh
module load python/intel/2.7.i
cd $HOME/VirtualRat/publication/
chmod 755 TestBlockTime.py
./TestBlockTime.py $SLURM_ARRAY_TASK_ID
