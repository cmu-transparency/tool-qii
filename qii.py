import pandas as pd
import numpy as np
import sklearn as skl
import numpy

import numpy.linalg
import sys
import time

from ml_util import *
from qii_lib import *

from sklearn.datasets import load_svmlight_file

### TODO:
### USe the random seed from command line for other things than model training.
###  - train/test splits
###  - iterations in the various qii computations

#def main():


args = get_arguments()
qii.record_counterfactuals = args.record_counterfactuals

#Read dataset
dataset = Dataset(args.dataset, sensitive=args.sensitive, target=args.target)
#if (args.erase_sensitive):
#  print 'Erasing sensitive'
#  dataset.delete_index(args.sensitive)

measure = args.measure
individual = args.individual

#Get column names
f_columns = dataset.num_data.columns
sup_ind = dataset.sup_ind

######### Begin Training Classifier ##########

cls, scaler, X_train, X_test, y_train, y_test, sens_train, sens_test = split_and_train_classifier(args, dataset)
print('End Training Classifier')
######### End Training Classifier ##########

measure_analytics(dataset, cls, X_test, y_test, sens_test)

t0 = time.time()

if measure == 'discrim':
    baseline = qii.discrim(numpy.array(X_test), cls, numpy.array(sens_test))
    discrim_inf = qii.discrim_influence(dataset, cls, X_test, sens_test)
    discrim_inf_series = pd.Series(discrim_inf, index = discrim_inf.keys())
    if (args.show_plot):
        plot_series_with_baseline(discrim_inf_series, args, 'Feature', 'QII on Group Disparity', baseline)

if measure == 'average-unary-individual':
    (average_local_inf, counterfactuals) = qii.average_local_influence(dataset, cls, X_test)
    average_local_inf_series = pd.Series(average_local_inf, index = average_local_inf.keys())
    if (args.show_plot):
        plot_series(average_local_inf_series, args, 'Feature', 'QII on Outcomes')

if measure == 'unary-individual':
    print individual
    x_individual = scaler.transform(dataset.num_data.ix[individual].reshape(1,-1))
    (average_local_inf, counterfactuals) = qii.unary_individual_influence(dataset, cls, x_individual, X_test)
    average_local_inf_series = pd.Series(average_local_inf, index = average_local_inf.keys())
    if (args.show_plot):
        plot_series(average_local_inf_series, args, 'Feature', 'QII on Outcomes')

if measure == 'banzhaf':
    print individual
    x_individual = scaler.transform(dataset.num_data.ix[individual])
    print dataset.num_data.ix[individual]

    banzhaf = qii.banzhaf_influence(dataset, cls, x_individual, X_test)
    banzhaf_series = pd.Series(banzhaf, index = banzhaf.keys())
    if (args.show_plot):
        plot_series(banzhaf_series, args, 'Feature', 'QII on Outcomes (Banzhaf)')

if measure == 'shapley':
    #print individual

    row_individual = dataset.num_data.ix[individual].reshape(1,-1)
    
    x_individual = scaler.transform(row_individual)
    
    #print dataset.num_data.ix[individual]

    shapley, counterfactuals = qii.shapley_influence(dataset, cls, x_individual, X_test)
    shapley_series = pd.Series(shapley, index = shapley.keys())
    if (args.show_plot):
        plot_series(shapley_series, args, 'Feature', 'QII on Outcomes (Shapley)')

t1 = time.time()
print (t1 - t0)







#if __name__ == '__main__':
#    main()


