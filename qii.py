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

cls, scaler, X_train, X_test, y_train, y_test, sens_train, sens_test = split_and_train_classifier(args.classifier, dataset)
print('End Training Classifier')
######### End Training Classifier ##########

measure_analytics(dataset, cls, X_test, y_test, sens_test)

t0 = time.time()

if measure == 'discrim-inf':
    baseline = qii.discrim(numpy.array(X_test), cls, numpy.array(sens_test))
    discrim_inf = qii.discrim_influence(dataset, cls, X_test, sens_test)
    discrim_inf_series = pd.Series(discrim_inf, index = discrim_inf.keys())
    if (args.show_plot):
        plot_series_with_baseline(discrim_inf_series, args, 'Feature', 'QII on Group Disparity', baseline)

if measure == 'average-local-inf':
    (average_local_inf, counterfactuals) = qii.average_local_influence(dataset, cls, X_test)
    average_local_inf_series = pd.Series(average_local_inf, index = average_local_inf.keys())
    if (args.show_plot):
        plot_series(average_local_inf_series, args, 'Feature', 'QII on Outcomes')

if measure == 'general-inf':
    average_local_inf = {}
    iters = 30
    y_pred = cls.predict(X_test)
    for sf in dataset.sup_ind:
        local_influence = numpy.zeros(y_pred.shape[0])
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for i in xrange(0, iters):
            X_inter = random_intervene(numpy.array(X_test), ls)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + y_pred_inter

        average_local_inf[sf] = (y_pred - local_influence/iters).mean()
        print('General Influence %s: %.3f' % (sf, average_local_inf[sf]))

    average_local_inf_series = pd.Series(average_local_inf, index = average_local_inf.keys())
    plot_series(average_local_inf_series, args, 'Feature', 'QII on Average Outcomes')

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


