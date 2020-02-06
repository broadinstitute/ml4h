.PHONY: install test

default: test

install:
	pipenv install --dev --skip-lock
test:
	pytest