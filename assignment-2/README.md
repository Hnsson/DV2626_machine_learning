# DV2626 Assignment 2: Wine Quality Classification
## Installation
You can run the project using these methods:

### Option 1 (Python venv)
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Option 2 (Install dependencies directly)
If you don't want a virtual environment:
```bash
pip install -r requirements.txt
```

### Option 3 (Use uv, fastest method)
If you have uv installed, you can run the script without manually installing dependencies:
```bash
uv run main.py
```
To install dependencies into the environment:
```bash
uv sync
```

## Running
Once dependencies are installed, simply run:
```bash
python main.py
```
Or, if using uv:
```bash
uv run main.py
```

