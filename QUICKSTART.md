# Awkward 1.0

### Dependencies
- cmake 
- gcc
- g++
- pytest


### Setup
- Clone the repo:
	`git clone --recursive https://github.com/scikit-hep/awkward-1.0.git`

- Build awkward:
	`python setup.py build`

- Test build:
	`pytest -vv tests`


### Contribution
- Branch off, create a pull request - PR# used to name new tests
- Clean build dir with:  
	- rm -rf **/*~ **/__pycache__ build dist *.egg-info awkward1/*.so **/*.pyc
	- Do this whenever:
		- CMakeLists.txt changes (cpp tests must be manually included, follow existing format)
