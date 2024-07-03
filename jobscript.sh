#!/bin/bash --login
#$ -cwd

#NVIDIA settings for CUDA 
#$ -l nvidia_v100=4
#$ -l nvidia_a100=4
#$ -pe smp.pe 8



# Load all required modules
module load libs/cuda/12.2.2 

# These env vars (without the SINGULARITY_) will be visible inside the image at runtime
export APPTAINER_HOME="$HOME"
export APPTAINER_LANG="$LANG"
# Make CSF scratch and your home dir visible to the container
export APPTAINER_BINDPATH="/scratch,/mnt"
# A GPU job on the CSF will have set $CUDA_VISIBLE_DEVICE, so test whether it is set or not (-n means "non-zero")
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
   # We are a GPU job. Set the special SINGULARITYENV_CUDA_VISIBLE_DEVICES to limit which GPUs the container can see.
   export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
   # Flag for the singularity command line
   NVIDIAFLAG=--nv
fi
# We use the 'sg' command to ensure the container is run with your own group id.
sg $GROUP -c "singularity run $NVIDIAFLAG my_container.sif arg1 arg2 ..."

mpirun -n $NSLOTS singularity exec name_of_container name_of_app_inside_container