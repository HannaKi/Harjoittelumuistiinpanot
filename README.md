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

## CSC computers 

*In Mahti full nodes are allocated for jobs, with the exception of interactive jobs*. 
Many options also work differently in Puhti and Mahti, so it is not advisable to copy scripts from Puhti to Mahti.

## Running a batch job

Always run a a test with test partition first. Remember to clear folders before you run the actual batch job.

```bash
sbatch MY_FILE.bash # aja batch job
squeue -u $USER # kysy jonossa olevat työt
gpuseff <JOBID> to get GPU efficiency statistics
```

To see the process during the training:
```bash
squeue -u $USER # to see the node id
ssh NODEID # go to the node
top # see the processes (exit with Ctrl + c)
nvidia-smi # see the GPUs in work
exit # to exit the node
```
If "Volatile GPU-Util" (percentages) is low increase batch size. 80-90 % is ok.

### Running sinteractive partition in Mahti

In Mahti you can only adjust the memory reserved by running an interactive job!

sinteractive --account project_200XXXX --time 48:00:00 --cores 6

module purge
module load XXX

python -m pip install --user XXXXX

No batch job script needed in interactive mode, just ryn your .py file.

### Running a serial job (Mahti) 

```bash
module purge
module load pytorch/1.8
cd transformers/
python -m pip install --user .
cd ..
python -m pip install --user -r requirements.txt
```
#### Contents of some Mahti batch job .sh file:

Test partition:

```bash
#!/bin/bash
#SBATCH --account=project_200XXXX
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
#SBATCH --account=Project_200XXXX

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

### sinteractive in Mahti

In Mahti, users can have several interactive batch job sessions in the interactive partition. Other partitions don't support interactive batch jobs. Each interactive session can reserve 1-8 cores, but the total number of reserved cores shared with all user sessions cannot exceed 8. Thus a user can have for example 4 interactive sessions with 2 cores or one 8 core session. Each core reserved will provide 1.875 GB of memory and the only way to increase the memory reservation is to increase the number of cores reserved. The maximum memory, provided by 8 cores, is 15 GB.

For example, an interactive session with 6 cores, 11,25 GiB of memory and 48 h running time using project project_2001234 can be launched with command:

`sinteractive --account project_2001234 --time 48:00:00 --cores 6`

#### Caches:

... are bad!

Before training remove this, if you have it: cd ~/.cache/huggingface/
and even better, set the path: https://huggingface.co/docs/datasets/installation.html

In the batch job sctipt file:

```bash
export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE="/scratch/project_2002820/hanna/bert_cache"
```
Should also be in the script: export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}:/scratch/project_2002820/hanna/tydiqa/hf_cache OR optionally give it as a param to the python code for training.

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

## Bash 

**Chmod**
chmod -R a+rX tydiqa

**Symbolic link**
In computing, a symbolic link (also symlink or soft link) is a term for any file that contains a reference to another file or directory in the form of an absolute or relative path and that affects pathname resolution
`ln -s /scratch/project_2002820/filip/paraphrases para-orig`

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


# Adapting the run_qa.py to paraphrase detection

1. 
in the training parameters set:
--pad_to_max_length False \

in the run_qa.py from line 121:

```bash
     pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
```
2. In the run_qa.py starting from line 334 comment out `truncation="only_second" if pad_on_right else "only_first"`

```bash
# Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            # truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

Should this be done with `def prepare_validation_features(examples)` also?!
