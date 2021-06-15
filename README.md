# Harjoittelumuistiinpanot TurkuNLP 2021

## Python Virtual Environment

`module purge` to clear the loaded modules.

`module load pytorch/1.6` load non Singularity torch.

And then create a virtual environment, activate it, and install the requirements

`python3 -m venv VENV-NAME --system-site-packages` this creates a virtual environment.
`source VENV-NAME/bin/activate` This activates the virtual environment, so after this line, you will be in the venv until you use the command `deactivate`

To install (in this case library "datasets" run: `python3 -m pip install datasets`. If pip version is too old, run: `pip install --upgrade pip`

To run HuggingFace example projects install from the source:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install .
cd ..
python -m pip install -r requirements.txt
```

```bash
sbatch slurm_train_qa.bash # aja batch job
squeue -u $USER # kysy jonossa olevat työt
gpuseff <JOBID> to get GPU efficiency statistics
```

## Singularity containers

If you run the stuff in a Singularity container no venv is needed.

`module purge` clear the loaded modules

`module load pytorch/1.8` Load a Singularity container (depends on the version if the module is a container or not)

To run HuggingFace example projects install from the source:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install --user . 
cd ..
```
Important!!! --user option allows user installations in the container!

Set the path if needed:
`export PYTHONPATH="${PYTHONPATH}:/users/kittihan/.local/bin"`

`python -m pip install --user -r requirements.txt` --user option again

To see the process during the training:
```bash
squeue -u $USER # to see the node id
ssh NODEID # go to the node
top # see the processes
nvidia-smi # see the GPUs in work
exit # to exit the node
```

kasvata batch size jos GPU hyödyntäminen alhainen

jos gpu muistia on jäljellä, kasvata batch size

tyhjennä hakemistot ennen kuin ajat varsinaisella 

## CSC computers 

Only Mahti GPU nodes have NVMe disk on compute nodes.
*Otherwise full nodes are allocated for jobs, with the exception of interactive jobs*, also see below. Many options also work differently in Puhti and Mahti, so it is not advisable to copy scripts from Puhti to Mahti

### Running a serial job (Mahti) 

```bash
module purge
module load pytorch/1.8
cd transformers/
python -m pip install --user .
cd ..
python -m pip install --user -r requirements.txt
```
### Contents of some Mahti batch job .sh file:

Test partition:

```bash
#!/bin/bash
#SBATCH --account=project_2002820
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load pytorch/1.8

srun python predict_squad.py
```
Mahti GPU batch job:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --partition=gpusmall
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=out_%A_%a.txt
#SBATCH --error=err_%A_%a.txt
#SBATCH --account=Project_2002820

module purge
module load pytorch/1.8 # if in a Puhti virtual env use pytorch/1.6 not Singularity /1.7+

# source /scratch/project_2002820/hanna/my_venv/bin/activate # activate venv, only in Puhti

# to not to cognest csc scratch:
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINDE_BERT_CACHE="/scratch/project_2002820/hanna/bert_cache"
export PYTORCH_TRANSFORMERS_CACHE="/scratch/project_2002820/hanna/bert_cache"

srun python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --output_dir /scratch/project_2002820/hanna/SQuAD2.0_results/
#  --overwrite_output_dir
```

To run the batc job: `sbatch FILE_NAME.sh`

### Batch job script parameters and commands

GPU resource, number of processors to use: `--gres=gpu:v100:<number_of_gpus_per_node>` 

#### Cache model:

In the batch job sctipt file:

`module load` 

```bash
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE="/scratch/project_2002820/hanna/bert_cache"
```
## Useful commands in CSC computers:

`module list` list loaded modules
`module purge` detach all the modules
`module spider`
`module load pytorch/1.6` load a module


Run your test in the test queue or in an interactive session directly from the command line

## Hints, tips and tricks for HPC

Look at an individual job in detail: `seff xxxxxx` 

Look at the jobs together: `sacct -j xxxxxx` or: `sacct -o jobname, jobid, reqmem, maxrss, timelimit, elapsed, state - j xxxxx`

Jos työ varaa 40 corea (kaikki) se vie myös koko noden muistin (silloin on ok varata myös noden muisti)

- Ohjelman asentaminen: jos haluat lisätä ympäristöön komennon, mene kansioon, ja luo polku sinne
- Muistin suhteen tähtää 1-2 gigan ekstraan. CPU ajassa 100 % tehokkuuteen
- Jos sbatchin jälkeen saat viestin kun kysyt squeue -u $USER "(Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)" --> 



## VIM

Like most Unix programs Vim can be suspended by pressing CTRL-Z. This stops Vim and takes you back to the shell it was started in. You can then do any other commands until you are bored with them. Then bring back Vim with the "fg" command.

```bash
CTRL-Z
{any sequence of shell commands}
fg
```

You are right back where you left Vim, nothing has changed.

https://vi.stackexchange.com/questions/16189/how-to-switch-between-buffer-and-terminal


## Virtual machine 

How to allocate more HD space: https://www.howtogeek.com/124622/how-to-enlarge-a-virtual-machines-disk-in-virtualbox-or-vmware/

## Interactive session with Jupyter notebook (not advisable? Does not work like this anyways...)

1. Start interactive session

   sinteractive --account project_XXXXXXX --time 02:00:00 --cores 1
   
2. Load suitable modules and start RStudio or Jupyter Notebook server
   
   module load pytorch/1.8
   pip install --user jupyter
   (Not sure if needed: export PATH=$PATH:~/.local/bin)
   NOPE: python3 -m notebook

3. Create SSH tunnel from your PC to Puhti compute node

4. Open RStudio or Jupyter Notebook in local web browser
