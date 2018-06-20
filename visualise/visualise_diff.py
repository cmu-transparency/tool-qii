from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from sklearn import tree
import numpy
import pandas

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

QII_SUM = 'ShapleySum'
ADULT_COLUMNS = ['Age', 'Workclass', 'Education', 'Education-Num', 'Marital Status', 'Occupation', 'Relationship',
                 'Race', 'Gender', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']


def create_explanation_graph(decision_tree, cluster_size, columns=None):
    class_names = [str(i) for i in range(0, cluster_size + 1)]
    dot_data = StringIO()
    export_graphviz(decision_tree, out_file=dot_data,
                    feature_names=columns,
                    filled=True, rounded=True, impurity=False,
                    special_characters=True, class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph


def cluster_dbscan(shapleys_df):
    model4 = DBSCAN(eps=0.1, min_samples=(shapleys_df.shape[0] // 20))
    clust_labels4 = model4.fit_predict(shapleys_df)
    return clust_labels4


def explain_dbscan(shapleys_df, cluster):
    print('NumClusters:', cluster.max())
    print('Cluster sizes: ', [(cluster == i).sum() for i in range(-1, cluster.max() + 1)])
    explainer = tree.DecisionTreeClassifier(max_leaf_nodes=8)
    explainer.fit(shapleys_df, cluster)

    return create_explanation_graph(explainer, cluster.max() + 1, list(shapleys_df))


def load_shapley_data(suffix):
    x_filename = "processed_data/x_samples%s.npy" % suffix
    y_filename = "processed_data/y_samples%s.npy" % suffix
    qii_filename = "processed_data/qii_samples%s.npy" % suffix
    inputs_raw = numpy.load(x_filename)
    predictions_raw = numpy.load(y_filename)
    shapley_values_raw = numpy.load(qii_filename)
    inputs = pandas.DataFrame.from_records(inputs_raw)
    shapley_values = pandas.DataFrame.from_records(shapley_values_raw, columns=ADULT_COLUMNS)
    predictions = pandas.DataFrame.from_records(predictions_raw, columns=["prediction"])
    return inputs, shapley_values, predictions


def __main__():
    x_df, qii_df, y_df = load_shapley_data("_10k_dbl_decision-forest")
    x_lr, qii_lr, y_lr = load_shapley_data("_10k_dbl_logistic")

    matches = []
    diffs = []
    for i in range(0, x_df.shape[0]):
        if y_df.iloc[i, 0] == y_lr.iloc[i, 0]:
            matches.append(i)
        else:
            diffs.append(i)

    print(len(matches))
    print(matches)
    print(len(diffs))
    print(diffs)

    qii_df_diffs_only = qii_df.iloc[diffs]
    x_df_diffs_only = x_df.iloc[diffs]
    cluster = cluster_dbscan(qii_df_diffs_only)

    graph = explain_dbscan(qii_df_diffs_only, cluster)
    graph.write_png("/tmp/graph_shapleys.png")

    mapper = tree.DecisionTreeClassifier(max_leaf_nodes=8)
    mapper.fit(x_df_diffs_only.values, cluster)
    graph = create_explanation_graph(mapper, cluster.max() + 1)
    graph.write_png("/tmp/graph_inputs.png")


__main__()
