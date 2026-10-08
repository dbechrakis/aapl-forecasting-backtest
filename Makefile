# Run the same checks as CI: `make check`. Override the interpreter with `make check PYTHON=python3.12`.
PYTHON ?= python
export PYTHONPATH := src:tests

.PHONY: install check lint evidence test data backtest forecast

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src scripts tests coursework --select E4,E7,E9,F

evidence:
	$(PYTHON) ci/verify_evidence.py

test:
	$(PYTHON) -m unittest discover -s tests -v

check: lint evidence test

data:
	$(PYTHON) scripts/download_data.py

backtest:
	$(PYTHON) scripts/run_backtest.py

forecast:
	$(PYTHON) scripts/forecast_latest.py
