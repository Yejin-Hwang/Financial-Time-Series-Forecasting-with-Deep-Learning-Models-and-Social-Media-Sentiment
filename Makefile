#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = financial-forecasting
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

## Lint using ruff
.PHONY: lint
lint:
	ruff format --check src/
	ruff check src/

## Format source code with ruff
.PHONY: format
format:
	ruff check src/ --fix
	ruff format src/

## Run tests
.PHONY: test
test:
	python -m pytest tests/ -v

## Train TFT model on TSLA data
.PHONY: train-tsla
train-tsla:
	$(PYTHON_INTERPRETER) -m src.models.train_tft --data-path data/raw/TSLA_full_features.csv --symbol TSLA

## Train TFT model on AAPL data  
.PHONY: train-aapl
train-aapl:
	$(PYTHON_INTERPRETER) -m src.models.train_tft --data-path data/raw/AAPL_full_features.csv --symbol AAPL

## Train TFT model on NVDA data
.PHONY: train-nvda
train-nvda:
	$(PYTHON_INTERPRETER) -m src.models.train_tft --data-path data/raw/NVDA_full_features.csv --symbol NVDA

## Create conda environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Start Jupyter Lab
.PHONY: jupyter
jupyter:
	jupyter lab notebooks/

## Generate project documentation
.PHONY: docs
docs:
	mkdocs serve

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -c "from src.data.data_loader import load_and_prepare_data; print('Data processing complete')"

## Quick demo
.PHONY: demo
demo: requirements
	$(PYTHON_INTERPRETER) -c "
from src.models.tft_model import TFTConfig, create_tft_model; 
from src.data.data_loader import load_and_prepare_data;
print('🚀 Running quick TFT demo...');
df = load_and_prepare_data('data/raw/TSLA_full_features.csv', 'TSLA');
config = TFTConfig(training_days=30, prediction_days=3, max_epochs=5);
model = create_tft_model(config);
training_dataset, _ = model.create_dataset(df);
tft_model = model.create_model();
print('✅ Demo setup complete!')
"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available commands:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)