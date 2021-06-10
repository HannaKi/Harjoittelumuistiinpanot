# Harjoittelumuistiinpanot
TurkuNLP 2021

## CSC Puhti

`module purge` to clear the loaded modules.

`module load pytorch/1.6` load non Singularity torch.

And then create a virtual environment, activate it, and install the requirements

`python3 -m venv VENV-NAME --system-site-packages` this creates a virtual environment.
`source VENV-NAME/bin/activate` This activates the virtual environment, so after this line, you will be in the venv until you use the command `deactivate`

To run HuggingFace example projects install from the source:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install .
cd to folder where requirements.txt is
python -m pip install -r requirements.txt
```

```bash
sbatch slurm_train_qa.bash # aja batch job
squeue -u $USER # kysy jonossa olevat työt
gpuseff <JOBID> to get GPU efficiency statistics
```

## CSC MAHTI

Run the stuff in a Singularity container! No venv needed.

`module purge` clear the loaded modules
`module load pytorch/1.8` Singularity container

To run HuggingFace example projects install from the source:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
python -m pip install --user . 
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


Run your test in the test queue or in an interactive session directly from the command line

## Useful commands in CSC computers:

`module list` list loaded modules
`module purge` detach all the modules
`module spider`


--gres=gpu:v100:<number_of_gpus_per_node>

## Hints, tips and tricks for HPC

Look at an individual job in detail: `seff xxxxxx` 

Look at the jobs together: `sacct -j xxxxxx` or: `sacct -o jobname, jobid, reqmem, maxrss, timelimit, elapsed, state - j xxxxx`

Jos työ varaa 40 corea (kaikki) se vie myös koko noden muistin (silloin on ok varata myös noden muisti)

- Ohjelman asentaminen: jos haluat lisätä ympäristöön komennon, mene kansioon, ja luo polku sinne
- Muistin suhteen tähtää 1-2 gigan ekstraan. CPU ajassa 100 % tehokkuuteen
- Jos sbatchin jälkeen saat viestin kun kysyt squeue -u $USER "(Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)" --> 

## Batch job scripts

`module load` jälkeen:

```bash
export TMPDIR=$LOCAL_SCRATCH

export PYTORCH_PRETRAINED_BERT_CACHE="/scratch/project_2000539/jenna/bert_cache"
```
## Interactive session with Jupyter notebook

1. Start interactive session

   sinteractive --account project_XXXXXXX --time 02:00:00 --cores 1
   
2. Load suitable modules and start RStudio or Jupyter Notebook server
   
   module load pytorch/1.8
   pip install --user jupyter
   start-jupyter-server

3. Create SSH tunnel from your PC to Puhti compute node

4. Open RStudio or Jupyter Notebook in local web browser





## Virtual machine 

How to allocate more HD space: https://www.howtogeek.com/124622/how-to-enlarge-a-virtual-machines-disk-in-virtualbox-or-vmware/







