# Quantitative Input Influence tool

This code was originally developed for Datta-Sen-Zick Oakland'16 but has evolved since. Presently
the tool requires python2 in some cases so if something is going wrong, make sure you run it with
python2 instead of python3.

- To install required python modules:

  ```make requirements```

- To try on the adult dataset run:

  ```python2 qii.py adult --show-plot```

- To see additional options, run:

  ```python2 qii.py -h```

# Optional arguments

  * `-h`, `--help` Show this help message and exit
  * `-m {average-unary-individual,unary-individual,discrim,banzhaf,shapley}` `--measure {...}` Quantity of interest
  * `-s SENSITIVE, --sensitive SENSITIVE` Sensitive field
  * `-t TARGET`, `--target TARGET` Target field
  * `-e`, `--erase-sensitive` Erase sensitive field from dataset
  * `-p`, `--show-plot`       Output plot as pdf
  * `-o`, `--output-pdf`      Output plot as pdf
  * `-c {logistic,svm,decision-tree,decision-forest}`, `--classifier {...}` Classifier(s) to use
  * `--max_depth MAX_DEPTH` Max depth for decision trees and forests
  * `--n_estimators N_ESTIMATORS` Number of trees for decision forests
  * `--seed SEED`  Random seed, auto seeded if not specified
  * `-a ACTIVE_ITERATIONS`, `--active-iterations ACTIVE_ITERATIONS` Active Learning Iterations
  * `-r`, `--record-counterfactuals` Store counterfactual pairs for causal analysis
  * `-i INDIVIDUAL`, `--individual INDIVIDUAL` Index for Individualized Transparency Report
  * `--batch_mode BATCH_MODE` Run in batch mode
  * `--batch_mode_samples BATCH_MODE_SAMPLES` Number of samples to compute.
  * `--output_suffix OUTPUT_SUFFIX` Output suffix for output in batch mode

# Currently supported datasets:

  * adult  : UCI Income dataset
  * iwpc   : Warfarin dosage
  * nlsy97 : Arrest prediction from the NLSY 97

# Currently supported measures:

  * discrim           : Unary QII on discrimination
  * average-unary-individual : Average unary QII
  * unary-individual  : Unary QII on individual outcome (use -i k) for kth individual
  * general-inf       : Influence on average classification
  * shapley           : Shapley QII (use -i k) for kth individual
  * banzhaf           : Banzhaf QII (use -i k) for kth individual
