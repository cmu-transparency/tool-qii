"""
Simple visualisation and analysis scripts for qii data.
Usage (from project root):
    python visualise/visualise.py
"""

import numpy
import pandas
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

QII_SUM = 'ShapleySum'
ADULT_COLUMNS = ['Age', 'Workclass', 'Education', 'Education-Num', 'Marital Status', 'Occupation', 'Relationship',
                 'Race', 'Gender', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']


def load_shapley_data():
    shapley_inputs_raw = numpy.load("processed_data/adult_samples_100.npy")
    shapley_values_raw = numpy.load("processed_data/adult_qii_values_100.npy")
    shapley_inputs = pandas.DataFrame.from_records(shapley_inputs_raw)
    shapley_values = pandas.DataFrame.from_records(shapley_values_raw, columns=ADULT_COLUMNS)
    return shapley_inputs, shapley_values


def add_sum_column(qii_df):
    qii_df_with_sum = qii_df.copy()
    qii_df_with_sum[QII_SUM] = qii_df.sum(axis = 1)
    return qii_df_with_sum

# Cluster data by classification value, and check stddev across individuals
# with the same classification.
def print_qii_stddev(qii_df):
    qii_df_with_sum = add_sum_column(qii_df)
    rows = qii_df_with_sum.shape[0]
    model4 = DBSCAN(eps=0.1, min_samples=rows // 16)
    cluster = model4.fit_predict(qii_df_with_sum.iloc[:,-1].reshape(-1,1))
    print ('NumClusters:', cluster.max())
    for i in range(-1, cluster.max() + 1):
        df_i = qii_df_with_sum[cluster == i]
        print ("Custer %d has size %d, qii stddev :%f" % (i,(cluster == i).sum() , df_i.std()[QII_SUM]))

def print_feature_rank_by_qii(qii_df):
    qii_rank = qii_df.abs().rank(axis = 1)
    qii_rank_avg = qii_rank.mean()
    qii_rank_std = qii_rank.std()
    for column in qii_rank.columns:
        print ("Feature %s has qii rank: (Average %f, Std: %f)" %
               (column, qii_rank_avg[column], qii_rank_std[column]))
        rank_distro = pandas.value_counts(qii_rank[column].values).sort_index()
        rank_distro.plot(kind='bar', title = column + ' Qii rank distribution', x = 'Rank', y = 'Count')
        plt.show()


def __main__():
    _, qii_df = load_shapley_data()

    print_qii_stddev(qii_df)
    print_feature_rank_by_qii(qii_df)

__main__()