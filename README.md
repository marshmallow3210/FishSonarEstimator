# FishSonarEstimator

## Overview  
FishSonarEstimator is a sonar image–based fish-count estimation system powered by the NVIDIA Agent Toolkit. It automatically processes incoming acoustic images through an AI agent workflow and returns accurate, real-time fish population counts.

## Dataset  
This project uses the publicly available:  
[*Underwater surveys of mullet schools (Mugil liza) with Adaptive Resolution Imaging Sonar*](https://doi.org/10.5281/zenodo.4751942)

## Project Structure
```
.
├── examples/estimator
│ ├── src/estimator
│ │ ├── estimator_function.py          # Tool registration & wrapper
│ │ ├── register.py                    # AIQ workflow registration
│ │ └── configs/config.yml             # Tool & workflow config
│ ├── app.py                           # Flask inference server
│ └── model.py                         # TensorFlow model definition
├── external/aiqtoolkit-opensource-ui  # Front-end UI code
├── requirements.txt
└── README.md
```

## Prerequisites  
- Python 3.11+  
- virtualenv (ubuntu)  
- Git & Git LFS  
- NVIDIA Agent Toolkit CLI (`pip install aiq-toolkit`)

## Installation & Setup  
```bash
# 1. Clone
git clone https://github.com/marshmallow3210/FishSonarEstimator.git
cd FishSonarEstimator

# 2. Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Frontend deps
cd external/aiqtoolkit-opensource-ui
npm install
```

## Configuration
Edit paths and your NVIDIA API key in `examples/estimator/src/estimator/configs/config.yml`

## Running the Project
1. Start the inference API
```bash
cd examples/estimator/
python app.py
```

2. Install the estimator tool package
```bash
uv pip install -e examples/estimator
```

3. Serve the AIQ workflow
```bash
aiq serve --config_file examples/estimator/src/estimator/configs/config.yml
```

4. Launch the UI
```bash
cd external/aiqtoolkit-opensource-ui/
npm run dev
```

5. Interact<br>
Open your browser and ask "現在有幾隻魚?" or "How many fish are there now?" in the chat!
