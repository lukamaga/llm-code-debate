# Running LLM Code Debate on VU MIF HPC

## Quick Start

```bash
# 1. Copy project to HPC
scp -r llm-code-debate/ USERNAME@hpc.mif.vu.lt:/scratch/lustre/home/USERNAME/

# 2. SSH into HPC (either login node works)
ssh USERNAME@hpc.mif.vu.lt
# or: ssh USERNAME@uosis.mif.vu.lt

# 3. Go to project directory
cd /scratch/lustre/home/$USER/llm-code-debate

# 4. Download Ollama Singularity image (one time only, ~2 GB)
cd hpc && singularity pull ollama_latest.sif docker://ollama/ollama && cd ..

# 5a. Run batch experiment
sbatch hpc/run_experiment.sh

# 5b. OR run web interface
sbatch hpc/run_web.sh
```

## Web Interface via SSH Tunnel

After `sbatch hpc/run_web.sh`:

1. Check which node the job is running on:
   ```bash
   squeue -u $USER
   # Look at the NODELIST column, e.g., "gpu01"
   ```

2. Check the log for the SSH tunnel command:
   ```bash
   cat logs/web_<JOBID>.log
   ```

3. On your Mac, open a new terminal and run:
   ```bash
   ssh -N -L 5050:gpu01:5050 USERNAME@hpc.mif.vu.lt
   ```
   Replace `gpu01` with the actual node name from step 1.

4. Open `http://localhost:5050` in your browser.

## Resource Limits

| Resource       | Limit                          |
|---------------|-------------------------------|
| GPU partition  | Max 48 hours per job           |
| GPU type       | NVIDIA V100 (8x per node, DGX-1) |
| GPU VRAM       | 32 GB per V100                 |
| Monthly quota  | 100 GPU-hours (MIF default)    |
| CPU quota      | 1000 CPU-hours/month           |
| RAM per core   | 12500 MB (gpu partition)       |
| Node RAM       | 512 GB (gpu nodes)             |

7B models fit in 32 GB VRAM. For 13B+ models, request 2 GPUs: `--gres=gpu:2`.

## Customizing Experiments

Edit variables at the top of `run_experiment.sh`:

```bash
MODELS="qwen2.5-coder:7b deepseek-coder:6.7b codellama:7b-instruct"
MAX_ROUNDS=3
TASK_DIRS="tasks/easy tasks/medium tasks/hard"
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch log output in real time
tail -f logs/debate_<JOBID>.log

# Cancel a job
scancel <JOBID>

# Check GPU usage quota (account format: USERNAME_mif)
sshare -l -A ${USER}_mif -p -o GrpTRESRaw,TRESRunMins
```

## Estimated GPU Time

| Scenario | Tasks | Models | Rounds | Est. GPU hours |
|----------|-------|--------|--------|---------------|
| Quick test | 3 easy | 2 × 7B | 2 | ~0.5h |
| Easy+Medium | 10 tasks | 3 × 7B | 3 | ~3-4h |
| Full experiment | 15+ tasks | 3 × 7B | 3 | ~6-8h |
| Solo baseline | 15+ tasks | 3 × 7B | 1 | ~2h |

Each 7B model generates a response in ~5-15s on V100. Budget: 100 GPU-hours/month for MIF students.

## Troubleshooting

**"Ollama failed to start"**: The GPU might not be available. Check `nvidia-smi` in an interactive session:
```bash
srun -p gpu --gres gpu --pty $SHELL
nvidia-smi
```

**Models downloading slowly**: Models are stored in `hpc/ollama_data/`. After the first pull, they persist between jobs. First run may take 10-15 min to download models.

**"Permission denied"**: Make sure scripts are executable:
```bash
chmod +x hpc/run_experiment.sh hpc/run_web.sh
```

**SSH tunnel not working**: Make sure you use the correct compute node name (from `squeue -u $USER`), not `hpc.mif.vu.lt`.

## File Structure on HPC

```
/scratch/lustre/home/USERNAME/llm-code-debate/
├── hpc/
│   ├── ollama_latest.sif    # Ollama container (after singularity pull)
│   ├── ollama_data/         # Downloaded models (persistent)
│   ├── run_experiment.sh    # Batch experiment script
│   ├── run_web.sh           # Web interface script
│   └── singularity.def      # Container definition (if building custom)
├── venv_hpc/                # Python venv (auto-created on first run)
├── logs/                    # SLURM job logs
├── results/                 # Experiment results
├── debate_results.db        # SQLite database with all results
├── src/                     # Application code
└── tasks/                   # Task definitions
```
