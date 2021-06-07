# Harjoittelumuistiinpanot
TurkuNLP 2021

## CSC Puhti

module purge # clear the loaded modules
module load pytorch/1.6
// and then create a virtual environment, activate it, and install the requirements

python3 -m venv VENV-NAME --system-site-packages # this creates a virtual environment
source VENV-NAME/bin/activate # this activates the virtual environment, so after this line, you will be in the venv until you use the comman `deactivate`

// To run HuggingFace example projects install from the source:
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install .
cd to folder where requirements.txt is
python -m pip install -r requirements.txt

sbatch slurm_train_qa.bash # aja batch job
squeue -u $USER # kysy jonossa olevat työt
gpuseff <JOBID> to get GPU efficiency statistics

Run your test in the test queue or in an interactive session directly from the command line

## Useful commands in CSC computers:
module list # list loaded modules
module purge # detach all the modules
module spider


--gres=gpu:v100:<number_of_gpus_per_node>

## CSC MAHTI

//Run the stuff in a Singularity container! No venv needed.

module purge # clear the loaded modules
module load pytorch/1.8 # Singularity container

// To run HuggingFace example projects install from the source:
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install --user . # Important!!! --user option allows user installations in the container

// Set the path if needed:
export PYTHONPATH="${PYTHONPATH}:/users/kittihan/.local/bin"

python -m pip install --user -r requirements.txt # option again

//to see the process during the training 
squeue -u $USER # to see the node id
ssh NODEID # go to the node
top # see the processes
nvidia-smi # see the GPUs in work
exit # to exit the node




