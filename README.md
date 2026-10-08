# ReProSeg
Representative Prototype-based Segmentation

## Setup

#### 1. Check if uv is installed:
```bash
uv --version
```
   
#### 2. install uv if needed:
- using pip:
```bash
pip install uv
```
- from Astral:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
   
#### 3. Install the dependencies using uv:
```bash
uv sync
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. The main configuration files are located in `config/hydra/`.

There are three entry points, one per scenario, each submitted as a Slurm job via a script under
`src/scripts/` (see [Slurm](#slurm) below). Training is normally run on its own; a trained
checkpoint is then visualized and/or evaluated separately.

- **train** (`src/scripts/train.sh`): trains ReProSeg, producing checkpoints under the run's log directory.
- **visualize** (`src/scripts/visualize.sh <run_dir>`): loads a trained checkpoint and renders prototype visualizations.
- **evaluate** (`src/scripts/evaluate.sh <checkpoint>`): loads a trained checkpoint and computes an interpretability metric (e.g. consistency score); works for ReProSeg or PPNet checkpoints.

Each has its own top-level config (`config/hydra/train.yaml`, `visualize.yaml`, `evaluate.yaml`), composed from shared config groups:

- **env**, **data**, **model**, **logging**: shared across all three scenarios
- **training**: train-only (epochs, optimizer, learning rates, resume)
- **visualization**: visualize-only (top-k prototypes per concept)
- **evaluate**: evaluate-only, selects a metric (e.g. `evaluate/consistency.yaml`)

### Running with Different Configurations

Hydra overrides are passed straight through to each script.

#### 1. Train with the default configuration
```bash
src/scripts/train.sh
```

#### 2. Custom root config file
```bash
src/scripts/train.sh --config-name=debug
```

#### 3. Override sub-configurations
```bash
# Use fast training configuration
src/scripts/train.sh training=fast

# Use a different data configuration
src/scripts/train.sh data=cityscapes
```

#### 4. Override individual parameters
```bash
# Change batch size and epochs
src/scripts/train.sh data.batch_size=8 training.epochs.total=500

# Change GPU ID and learning rate
src/scripts/train.sh env.gpu_id=0 training.learning_rates.classifier=0.01
```

#### 5. Visualize or evaluate a trained checkpoint
```bash
# Visualize prototypes for a trained checkpoint; reuses its exact training config
src/scripts/visualize.sh <run_dir>

# Compute the consistency score for a checkpoint (ReProSeg or PPNet)
src/scripts/evaluate.sh <checkpoint> evaluate=consistency data=pascal_voc
```

### Environment Variables

The configuration system also supports environment variables:
- Set `LOG_ROOT` environment variable to customize the log output directory
- Neural Network Intelligence integration is supported via `NNI_TRIAL_JOB_ID`

## Slurm

Each script submits a Slurm job via the generic `src/scripts/_submit.sh` (just `#SBATCH` resource
directives + `uv run python "$@"`). Override resources on the sbatch command line if needed, e.g.:
```bash
sbatch --gres=gpu:0 --mem=8G src/scripts/_submit.sh -m evaluate ...
```

One-off sweeps (e.g. over `--quantile`) are plain shell loops over `sbatch src/scripts/_submit.sh -m evaluate ...`.

## HPO

Hyper parameter optimization is done using Neural Network Intelligence (nni).

### Running HPO

Run NNI from the project root directory:
```bash
nnictl create --config src/config/nni/nni_config.yaml
```

This will:
- Execute 20 trials using TPE (Tree-structured Parzen Estimator) optimization
- Run one trial at a time (`trialConcurrency: 1`)
- Use median stopping to terminate poor-performing trials after step 70
- Optimize to maximize the target metric

### Managing HPO Experiments

```bash
# View experiment status
nnictl view

# Stop the experiment
nnictl stop <experiment_id>

# View web UI (opens in browser)
nnictl webui url
```

### Configuration

The HPO search space is defined in [src/config/nni/search_space.json](src/config/nni/search_space.json). Modify this file to adjust which hyperparameters to optimize and their ranges.

## Running Tests

Run the dataset unit tests:
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest <test_file_name>

# Run specific test method in a file
uv run pytest <test_file_name>::<TestClassName>::<test_method_name>

# Run with coverage
uv run pytest --cov=src --cov-report=term
```

## Code Formatting and Type Checking

```bash
# Code formatting:
uv run ruff format [--check]
# with --check, show potential changes without applying them.
 
# Checking PEP conventions:
uv run ruff check [--fix]
# when the fix argument is provided, the program automatically corrects the errors it can. Typically, it cannot fix overly long lines, but if you run formatting beforehand, such errors should not occur.
 
# Type checking (src - source code folder):
uv run mypy src
```
