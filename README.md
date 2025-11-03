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

The project uses [Hydra](https://hydra.cc/) for configuration management. The main configuration files are located in `src/config/yaml/`.

### Configuration Structure

The base configuration is defined in `src/config/yaml/config.yaml`, which uses Hydra's composition feature to combine multiple sub-configurations:

- **data**: Dataset and dataloader settings (`src/config/yaml/data/`)
- **model**: Model architecture and parameters (`src/config/yaml/model/`)  
- **training**: Training parameters and epochs (`src/config/yaml/training/`)
- **logging**: Logging configuration (`src/config/yaml/logging/`)

### Running with Different Configurations

#### 1. Default Configuration
Run with the default configuration defined in `config.yaml`:
```bash
uv run python src/scripts/run.py
```

#### 2. Custom Root Config File
Different root configuration file can be used, for example debug configurations:
```bash
uv run python src/scripts/run.py --config-name=debug
```

#### 3. Override Sub-configurations
Override specific configuration groups from the default root config:
```bash
# Use fast training configuration
uv run python src/scripts/run.py training=fast

# Use a different data configuration
uv run python src/scripts/run.py data=other_dataset

# Override multiple configuration groups
uv run python src/scripts/run.py training=fast model=custom
```

#### 4. Override Individual Parameters
Override specific configuration parameters:
```bash
# Change batch size and epochs
uv run python src/scripts/run.py data.batch_size=8 training.epochs.total=500

# Change GPU ID and learning rate
uv run python src/scripts/run.py env.gpu_id=0 training.learning_rates.classifier=0.01

# Skip training and only run visualization
uv run python src/scripts/run.py training.skip_training=true visualization.generate_explanations=true

# Enable consistency score calculation
uv run python src/scripts/run.py evaluation.consistency_score.calculate=true
```

#### 5. Complex Configuration Override Examples
```bash
# Fast training with custom batch size and GPU
uv run python src/scripts/run.py training=fast data.batch_size=4 env.gpu_id=0

# Custom training configuration with model parameters
uv run python src/scripts/run.py \
  training.epochs.total=200 \
  training.epochs.pretrain=50 \
  model.num_prototypes=64 \
  model.loss_weights.classification=5.0

# Run only evaluation without training
uv run python src/scripts/run.py \
  training.skip_training=true \
  visualization.generate_explanations=false \
  evaluation.consistency_score.calculate=true
```

### Environment Variables

The configuration system also supports environment variables:
- Set `LOG_ROOT` environment variable to customize the log output directory
- NNI (Neural Network Intelligence) integration is supported via `NNI_TRIAL_JOB_ID`

### Configuration Help

To see all available configuration options and their current values:
```bash
uv run python src/scripts/run.py --help
```

To print the complete configuration that would be used:
```bash
uv run python src/scripts/run.py --cfg job
```

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
