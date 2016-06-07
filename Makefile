# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= py.test
PROJ_NAME ?= learning

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

inplace:
	$(PYTHON) setup.py build_ext -i

test: inplace
	$(PYTEST) -s -v --durations=10 --doctest-modules hmmlearn

trailing-spaces:
	find $(PROJ_NAME) -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find $(PROJ_NAME) -name "*.pyx" | xargs $(CYTHON)

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 hmmlearn | grep -v __init__ | grep -v external
	pylint -E -i y hmmlearn/ -d E1103,E0611,E1101
