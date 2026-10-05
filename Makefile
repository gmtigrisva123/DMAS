PY ?= python3
export PYTHONPATH := src

.PHONY: help session demo diagnose experiment bench test lint clean install

help:
	@$(PY) -m deductive_mas --help

install:
	$(PY) -m pip install -e ".[dev]"

session:
	$(PY) -m deductive_mas session

demo:
	$(PY) -m deductive_mas demo

diagnose:
	$(PY) -m deductive_mas diagnose --problem binary_search_lower_bound --submission bs_off_by_one

experiment:
	$(PY) -m deductive_mas experiment --students 120 --seed 20260909

bench:
	$(PY) -m deductive_mas bench

test:
	$(PY) -m pytest

clean:
	rm -rf .pytest_cache **/__pycache__ artifacts .dmas_cache
