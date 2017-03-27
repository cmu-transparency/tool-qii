pylint: *.py
	pylint -f parseable -j 4 *.py

test:
	python qii.py -m average-unary-individual final.csv
	python qii.py -m unary-individual final.csv
	python qii.py -m discrim final.csv
	python qii.py -m banzhaf final.csv
	python qii.py -m shapley final.csv

clean:
	rm -Rf *.pyc
	rm -Rf *~
