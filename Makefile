OS   := $(shell uname)
ifeq ($(OS), Darwin)
  TIME := time -l
else
  TIME := time -v
endif

pylint: *.py
	pylint -f parseable -j 4 *.py

test-shapley:
	$(TIME) python qii.py -m shapley final.csv

test:
	python qii.py -m average-unary-individual final.csv
	python qii.py -m unary-individual final.csv
	python qii.py -m discrim final.csv
	python qii.py -m banzhaf final.csv
	python qii.py -m shapley final.csv

clean:
	rm -Rf *.pyc
	rm -Rf *~
