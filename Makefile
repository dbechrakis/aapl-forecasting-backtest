# Run the same checks as CI: `make check`. Override the interpreter with `make check PYTHON=python3.12`.
PYTHON ?= python

.PHONY: install check lint evidence test

install:
	$(PYTHON) -m pip install -r requirements.txt

evidence:
	$(PYTHON) ci/verify_evidence.py

test:
	$(PYTHON) -m unittest discover -s tests -v

check: evidence test
