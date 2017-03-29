OS   := $(shell uname)
ifeq ($(OS), Darwin)
  TIME := /usr/bin/time -l
else
  TIME := /usr/bin/time -v
endif

pylint: *.py
	pylint -f parseable -j 4 *.py

test-shapley:
	$(TIME) python qii.py -m shapley final.csv

test:
	$(TIME) python qii.py -m average-unary-individual final.csv
	$(TIME) python qii.py -m unary-individual final.csv
	$(TIME) python qii.py -m discrim final.csv
	$(TIME) python qii.py -m banzhaf final.csv
	$(TIME) python qii.py -m shapley final.csv

clean:
	rm -Rf *.pyc
	rm -Rf *~
