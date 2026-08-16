.PHONY: help setup data explore train tune simulate monitor multi bayesian experiment all test api dashboard docker clean criteo criteo5

PYTHON ?= python3
export PYTHONPATH := src

help:
	@echo "make setup      install dependencies"
	@echo "make data       download the Hillstrom dataset"
	@echo "make all        full pipeline: explore -> train -> simulate -> monitor -> multi -> bayesian -> experiment"
	@echo "make test       run the test suite"
	@echo "make api        serve the FastAPI app on :8000"
	@echo "make dashboard  serve the Streamlit dashboard on :8501"
	@echo "make bayesian   posterior over treatment effects + calibration check"
	@echo "make experiment size the live A/B test that would validate the policy"
	@echo "make criteo     scale-up run on the full Criteo dataset (14M rows, ~18 min)"
	@echo "make criteo5    five-way comparison incl. causal forest, 20% sample (~24 min)"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/00_download_data.py --dataset hillstrom

explore:
	$(PYTHON) scripts/01_explore.py --dataset hillstrom

train:
	$(PYTHON) scripts/02_train.py --dataset hillstrom

tune:
	$(PYTHON) scripts/02a_tune.py --dataset hillstrom

simulate:
	$(PYTHON) scripts/03_simulate.py --dataset hillstrom

monitor:
	$(PYTHON) scripts/04_monitor.py --dataset hillstrom

multi:
	$(PYTHON) scripts/05_multi_treatment.py

bayesian:
	$(PYTHON) scripts/06_bayesian.py

experiment:
	$(PYTHON) scripts/07_experiment_design.py

all: data explore train simulate monitor multi bayesian experiment

# Full 13,979,592 rows: ~460 MB download, then ~18 min of compute at ~1.7 GB peak
# RSS. The causal forest is excluded deliberately — at 224.7s on a 10% sample it
# projects to hours across CV here, and its memory footprint does not fit in 8 GB.
# For the five-way comparison including the causal forest, use `make criteo5`.
criteo:
	$(PYTHON) scripts/00_download_data.py --dataset criteo
	$(PYTHON) scripts/02_train.py --dataset criteo --no-causal-forest --n-boot 50 --cv-splits 3 --cv-repeats 1
	$(PYTHON) scripts/03_simulate.py --dataset criteo --n-boot 50
	$(PYTHON) scripts/04_monitor.py --dataset criteo
	$(PYTHON) scripts/01_explore.py --dataset criteo

# Five-way comparison at the largest sample the causal forest fits on. 20% is
# not a guess: 40% was measured at a 17.5 GB peak footprint and was killed after
# 110 minutes of thrashing on this 8 GB machine. ~24 min, 4.1 GB peak RSS.
criteo5:
	$(PYTHON) scripts/02_train.py --dataset criteo --sample-frac 0.20 --n-boot 50 --cv-splits 3 --cv-repeats 1

test:
	$(PYTHON) -m pytest tests/ -v

api:
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

docker:
	docker compose up --build

clean:
	rm -rf reports/figures/*.png reports/*.json reports/*.jsonl models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} +
