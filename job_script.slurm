#!/bin/bash

## Job Name
#SBATCH --job-name=zdeblick_optimizer

## Allocation Definition

## group name and partition name of the nodes.
## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).
#SBATCH --account=stf
#SBATCH --partition=stf

##  Resources

## Number of Nodes to reserve. One will suffice unless you are doing some parallel computing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12

## Amount of time you need the node for (hrs:mins:seconds). Do not specify a walltime substantially more than your job needs.
#SBATCH --time=240:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.
## For mox, --mem may be more than 100G depending on the memory of your nodes.
## For ikt, --mem may be 58G or more depending on the memory of your nodes.

## See above section on "Specifying memory" for choices for --mem.
#SBATCH --mem=120G

## Specify the working directory for this job
#SBATCH --chdir=/gscratch/stf/zdeblick/BrainConstraints

## Specify the directory where STDOUT and STDERR files are stored
##SBATCH --output=/gscratch/stf/zdeblick/output_files/

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zdeblick@uw.edu

## export all your environment variables to the batch job session
##SBATCH --export=all


## ======== PUT COMMANDS HERE ==========================================================

## LOAD MODULES
module load anaconda3_5.3


## My Program
source activate env
python3 gmmglm_script_clean.py
