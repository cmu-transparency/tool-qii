""" QII mesurement script

author: mostly Shayak

"""

import time

import pandas  as pd
import numpy

import numpy.linalg

from ml_util import split_and_train_classifier, get_arguments, \
     Dataset, measure_analytics, \
     plot_series_with_baseline, plot_series

import qii_lib

def __main__():
    args = get_arguments()
    qii_lib.record_counterfactuals = args.record_counterfactuals

    #Read dataset
    dataset = Dataset(args.dataset, sensitive=args.sensitive, target=args.target)
    #Get column names
    #f_columns = dataset.num_data.columns
    #sup_ind = dataset.sup_ind

    ######### Begin Training Classifier ##########

    dat = split_and_train_classifier(args, dataset)

    print('End Training Classifier')
    ######### End Training Classifier ##########

    measure_analytics(dataset, dat.cls, dat.x_test, dat.y_test, dat.sens_test)

    t_start = time.time()

    measures = {'discrim': eval_discrim,
                'average-unary-individual': eval_average_unary_individual,
                'unary-individual': eval_unary_individual,
                'banzhaf': eval_banzhaf,
                'shapley': eval_shapley}

    if args.measure in measures:
        measures[args.measure](dataset, args, dat)
    else:
        raise ValueError("Unknown measure %s" % args.measure)

    t_end = time.time()

    print(t_end - t_start)

def eval_discrim(dataset, args, dat):
    """ Discrimination metric """

    baseline = qii_lib.discrim(numpy.array(dat.x_test), dat.cls, numpy.array(dat.sens_test))
    discrim_inf = qii_lib.discrim_influence(dataset, dat.cls, dat.x_test, dat.sens_test)
    discrim_inf_series = pd.Series(discrim_inf, index=list(discrim_inf.keys()))
    if args.show_plot:
        plot_series_with_baseline(
            discrim_inf_series, args,
            'Feature', 'QII on Group Disparity',
            baseline)

def eval_average_unary_individual(dataset, args, dat):
    """ Unary QII averaged over all individuals. """

    average_local_inf, _ = qii_lib.average_local_influence(
        dataset, dat.cls, dat.x_test)
    average_local_inf_series = pd.Series(average_local_inf,
                                         index=list(average_local_inf.keys()))
    if args.show_plot:
        plot_series(average_local_inf_series, args,
                    'Feature', 'QII on Outcomes')

def eval_unary_individual(dataset, args, dat):
    """ Unary QII. """

    x_individual = dat.scaler.transform(dataset.num_data.ix[args.individual].reshape(1, -1))
    average_local_inf, _ = qii_lib.unary_individual_influence(
        dataset, dat.cls, x_individual, dat.x_test)
    average_local_inf_series = pd.Series(
        average_local_inf, index=list(average_local_inf.keys()))
    if args.show_plot:
        plot_series(average_local_inf_series, args,
                    'Feature', 'QII on Outcomes')

def eval_banzhaf(dataset, args, dat):
    """ Banzhaf metric. """

    x_individual = dat.scaler.transform(dataset.num_data.ix[args.individual])

    banzhaf = qii_lib.banzhaf_influence(dataset, dat.cls, x_individual, dat.x_test)
    banzhaf_series = pd.Series(banzhaf, index=list(banzhaf.keys()))
    if args.show_plot:
        plot_series(banzhaf_series, args, 'Feature', 'QII on Outcomes (Banzhaf)')

def eval_shapley(dataset, args, dat):
    """ Shapley metric. """

    row_individual = dataset.num_data.ix[args.individual].reshape(1, -1)

    x_individual = dat.scaler.transform(row_individual)

    #shapley, _ = qii_lib.shapley_influence(dataset, dat.cls, x_individual, dat.x_test)
    shapley = qii_lib.shapley_influence_cached(dataset, dat.cls, x_individual, dat.x_test)
    print (shapley)
    shapley_series = pd.Series(shapley, index=list(shapley.keys()))
    if args.show_plot:
        plot_series(shapley_series, args, 'Feature', 'QII on Outcomes (Shapley)')

__main__()
