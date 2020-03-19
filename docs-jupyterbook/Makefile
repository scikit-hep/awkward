E.PHONY: help clean book

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  clean       to clean out site build files"
	@echo "  book        to convert the content/ folder into Jekyll markdown in _build/"
	@echo "  exec-book   to execute the content/ folder and put Jekyll markdown in _build/"

clean:
	python scripts/clean.py

book:
	jupyter-book build --execute ./

exec-book:
	jupyter-book build ./
