""" QII mesurement script

author: mostly Shayak

"""

import time

import pandas  as pd
import numpy
import pickle
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

    print ('End Training Classifier')
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

    print (t_end - t_start)

def eval_discrim(dataset, args, dat):
    """ Discrimination metric """

    baseline = qii_lib.discrim(numpy.array(dat.x_test), dat.cls, numpy.array(dat.sens_test))
    discrim_inf = qii_lib.discrim_influence(dataset, dat.cls, dat.x_test, dat.sens_test)
    discrim_inf_series = pd.Series(discrim_inf, index=discrim_inf.keys())
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
                                         index=average_local_inf.keys())
    if args.show_plot or args.output_pdf:
        plot_series(average_local_inf_series, args,
                    'Feature', 'QII on Outcomes')

def eval_unary_individual(dataset, args, dat):
    """ Unary QII. """

    x_individual = dat.scaler.transform(dataset.num_data.ix[args.individual].reshape(1, -1))
    average_local_inf, _ = qii_lib.unary_individual_influence(
        dataset, dat.cls, x_individual, dat.x_test)
    average_local_inf_series = pd.Series(
        average_local_inf, index=average_local_inf.keys())
    if args.show_plot or args.output_pdf:
        plot_series(average_local_inf_series, args,
                    'Feature', 'QII on Outcomes')

def eval_banzhaf(dataset, args, dat):
    """ Banzhaf metric. """

    x_individual = dat.scaler.transform(dataset.num_data.ix[args.individual])

    banzhaf = qii_lib.banzhaf_influence(dataset, dat.cls, x_individual, dat.x_test)
    banzhaf_series = pd.Series(banzhaf, index=banzhaf.keys())
    if args.show_plot or args.output_pdf:
        plot_series(banzhaf_series, args, 'Feature', 'QII on Outcomes (Banzhaf)')

def eval_shapley(dataset, args, dat):
    """ Shapley metric. """

    if (args.batch_mode):
        eval_shapley_batch(dataset, args, dat)
        return

    row_individual = dataset.num_data.ix[args.individual].reshape(1, -1)

    x_individual = dat.scaler.transform(row_individual)

    shapley, _ = qii_lib.shapley_influence(dataset, dat.cls, x_individual, dat.x_test)
    print (shapley)
    shapley_series = pd.Series(shapley, index=shapley.keys())
    if args.show_plot or args.output_pdf:
        plot_series(shapley_series, args, 'Feature', 'QII on Outcomes (Shapley)')


def eval_shapley_batch(dataset, args, dat):
    super_indices = list(dataset.sup_ind.keys())
    rowsize = dataset.num_data.shape[0]
    learning_samples = args.batch_mode_samples
    shapley_saved = numpy.zeros((learning_samples, len(super_indices)))
    x_samples = numpy.zeros(((learning_samples, dataset.num_data.shape[1])))
    time_last = time.time()
    for i in range(0, learning_samples):
        if i % 20 == 0:
            time_new = time.time()
            print ('Index:', i)
            print ('Time:', time_new - time_last)
            time_last = time_new
        idx = i * (rowsize // learning_samples)
        row_individual = dataset.num_data.ix[idx].reshape(1, -1)

        x_individual = dat.scaler.transform(row_individual)
        x_samples[i] = x_individual

        shapley, _ = qii_lib.shapley_influence(dataset, dat.cls, x_individual, dat.x_test)
        shapley_series = pd.Series(shapley, index=shapley.keys())
        for j in range(0, len(super_indices)):
            shapley_saved[i][j] = shapley_series[super_indices[j]]

    suffix = args.output_suffix
    x_filename = getfilename("processed_data/", "x_samples", suffix)
    qii_filename = getfilename("processed_data/", "qii_samples", suffix)
    args_filename = getfilename("processed_data/", "args", suffix)

    numpy.save(x_filename, x_samples)
    numpy.save(qii_filename, shapley_saved)
    with open(args_filename, 'wb') as pickle_file:
        pickle.dump(args, pickle_file, 0)

def getfilename(prefix, name, suffix):
    return prefix + name + suffix

__main__()
