PYTHON = python2
PIP = pip2
LINT = pylint

requirements:
	$(PIP) install pandas numpy sklearn matplotlib arff argparse statsmodels

pylint: *.py
	$(LINT) -f parseable -j 4 *.py

clean:
	rm -Rf *.pyc
	rm -Rf *~
