# ReProSeg
Representative Prototype-based Segmentation

## Setup

1. Check if uv is installed:
   ```bash
   uv --version
   ```
   
2. If not installed, install uv:
   - Using pip:
     ```bash
     pip install uv
     ```
   - Or from Astral:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
     
3. Navigate to the `src/ReProSeg` directory:
   ```bash
   cd src/ReProSeg
   ```
   
4. Install the dependencies using uv:
   ```bash
   uv sync
   ```

## Running Training

1. From the `src/ReProSeg` directory, run the training script:
   ```bash
   uv run python src/scripts/run.py
   ```

2. You can also pass additional arguments to customize the training. For example:
   ```bash
   uv run python src/scripts/run.py --dataset CityScapes --epochs 50 --batch_size 32
   ```

3. To see all available options, run:
   ```bash
   uv run python src/scripts/run.py --help
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