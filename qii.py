""" QII mesurement script

author: mostly Shayak

"""
import pdb
import time
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy.linalg
from matplotlib.backends.backend_pdf import PdfPages

from ml_util import split_and_train_classifier, get_arguments, \
	Dataset, measure_analytics, \
	plot_series_with_baseline, plot_series

import qii_lib


def __main__():
	args = get_arguments()
	qii_lib.record_counterfactuals = args.record_counterfactuals

	# Read dataset
	dataset = Dataset(args.dataset, sensitive=args.sensitive, target=args.target)
	# Get column names
	# f_columns = dataset.num_data.columns
	# sup_ind = dataset.sup_ind

	######### Begin Training Classifier ##########

	dat = split_and_train_classifier(args, dataset)

	print 'End Training Classifier'
	######### End Training Classifier ##########

	measure_analytics(dataset, dat.cls, dat.x_test, dat.y_test, dat.sens_test)

	t_start = time.time()

	measures = {'discrim': eval_discrim,
	            'average-unary-individual': eval_average_unary_individual,
	            'unary-individual': eval_unary_individual,
	            'banzhaf': eval_banzhaf,
	            'shapley': eval_shapley,
	            'average-unary-class': eval_class_average_unary}

	if args.measure in measures:
		measures[args.measure](dataset, args, dat)
	else:
		raise ValueError("Unknown measure %s" % args.measure)

	t_end = time.time()

	print t_end - t_start


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
	top_40 = average_local_inf_series.sort_values(ascending=False).head(40)
	if args.show_plot or args.output_pdf:
		plot_series(top_40, args,
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

	row_individual = dataset.num_data.ix[args.individual].reshape(1, -1)

	x_individual = dat.scaler.transform(row_individual)

	shapley, _ = qii_lib.shapley_influence(dataset, dat.cls, x_individual, dat.x_test)
	shapley_series = pd.Series(shapley, index=shapley.keys())
	if args.show_plot or args.output_pdf:
		plot_series(shapley_series, args, 'Feature', 'QII on Outcomes (Shapley)')


def eval_class_average_unary(dataset, args, dat):
	""" Unary QII averaged over all individuals for a particular class """
	average_local_inf, _ = qii_lib.average_local_class_influence(
		dataset, dat.cls, dat.x_test, dat.x_target_class)
	average_local_inf_series = pd.Series(average_local_inf,
	                                     index=average_local_inf.keys())
	top_40 = average_local_inf_series.sort_values(ascending=False).head(40)
	if args.show_plot or args.output_pdf:
		plot_series(top_40, args,
		            'Feature', 'QII on Outcomes')
	top_5 = average_local_inf_series.sort_values(ascending=False).head(5)
	get_feature_variation_plots(top_5, dataset, args, dat)


def get_feature_variation_plots(features_list, dataset, args, dat):
	def plot_histogram(dataframe):
		data = dataframe.copy()
		data = data.drop(['feature', 'class'], axis=1)
		data = data.set_index('bin_edges')
		data.hist()
		del data

	x_test = dat.x_test.reset_index(drop=True)
	y_test = dat.y_test.reset_index(drop=True)
	temp = x_test.copy()
	temp['class'] = y_test
	features = numpy.array(features_list.keys())
	for feature in features:
		plt.figure()
		# bins = numpy.unique(temp[feature])
		for class_index, class_group in temp.groupby(['class']):
			# plt.hist(class_group[feature], bins=bins, label=str(class_index))
			plt.hist(class_group[feature], label=str(class_index))
		plt.legend(loc='best')
		plt.title('Combined Histogram ' + str(feature))
		if args.output_pdf:
			pp = PdfPages('Combined Histogram-' + str(feature) + '-'+ args.classifier + '.pdf')
			print ('Writing to Combined Histogram-' + str(feature) + '-'+ args.classifier + '.pdf')
			pp.savefig(bbox_inches='tight')
			pp.close()
		if args.show_plot:
			plt.show()



	feature_variations = pd.DataFrame()
	for cls in dat.y_test.unique():
		x_target_class = x_test[y_test == cls]
		feature_variations = feature_variations.append(qii_lib.get_feature_variations(features_list,
		                                                                              dataset, dat.cls, dat.x_test,
		                                                                              x_target_class, cls))

		# features = numpy.array(features_list.keys())
		for feature in features:
			plt.figure()
			x_target_class[feature].hist()
			plt.title(str(feature) + '-' + 'class_' + str(cls))
			if args.output_pdf:
				pp = PdfPages('Histogram-' + str(feature) + '-' + 'class_' + str(cls) + args.classifier + '.pdf')
				print ('Writing to Histogram-' + str(feature) + '-' + 'class_' + str(cls) + args.classifier + '.pdf')
				pp.savefig(bbox_inches='tight')
				pp.close()
			if args.show_plot:
				plt.show()

	for index, group in feature_variations.groupby(['feature']):
		plt.figure()
		for class_index, class_group in group.groupby(['class']):
			plt.plot(class_group['bin_edges'], class_group['influences'], label=class_index)
		plt.legend(loc='best')
		plt.title(index)
		if args.output_pdf:
			pp = PdfPages('figure-' + index + '-' + args.classifier + '.pdf')
			print ('Writing to figure-' + index + '-' + args.classifier + '.pdf')
			pp.savefig(bbox_inches='tight')
			pp.close()
		if args.show_plot:
			plt.show()


__main__()
