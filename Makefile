SHELL := bash
.ONESHELL:

ifeq ($(origin .RECIPEPREFIX), undefined)
> $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

.PHONY: help, \
> setup, \
> lint_staged \
> lint_all

help:
> @echo
> @echo "ML4CVD Makefile commands"
> @echo
> @echo "setup       -- creates env and sets up git commit hooks."
> @echo
> @echo "lint_staged -- lints files staged for commit"
> @echo
> @echo "lint_all    -- lints all files"
> @echo

setup:
> @echo Setting up the repo...
> @conda env create -f .pre-commit-env.yml
> @conda run -n ml4cvd pre-commit install

lint_staged:
> @echo Running hook with staged files...
> @pre-commit run

lint_all:
> @echo Running hook on all files...
> @pre-commit run --all-files
