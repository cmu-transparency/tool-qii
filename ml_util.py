import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import LabelEncoder
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy
import numpy.random
import arff

import numpy.linalg
import sys
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import time

from os.path import exists


from qii_lib import *


#labelfont = {'fontname':'Times New Roman', 'size':15}
labelfont = {}
#hfont = {'fontname':'Helvetica'}

def get_column_index(data, cname):
    try:
        idx = data.columns.get_loc(cname)
    except Exception as e:
        raise ValueError("Unknown column %s" % cname)
    
    return idx

def encode_nominal(col):
    if col.dtype == object:
        return LabelEncoder().fit_transform(col)
    else:
        return col

import argparse
class Dataset(object):
    """
    Class that holds a dataset.
    Each dataset has its own quirks and needs some special processing
    to get to the point where we need it to.
    An important part of this is the treatment of categorical variables.
    Translating from categorical variables to regular variables leads
    to one feature getting mapped to k-1 features via one-hot encoding.
    However, while generating transparency reports we need to treat
    all of these low level numerical features as one feature. Therefore,
    we have a super index (sup_ind) that is a dict that maps from
    high level features to low level features.
    To add a new dataset create a choose a new name and create a case for it.
    Attributes:
        name: A string representing the name of the dataset
        original_data: Dataset with categorical variables.
        num_data: Dataset with cleaned numeric values
        sup_ind: Super Index containing a dict which maps from original
            feature to list of dummy features
        target_ix: Name of target index
        sensitive_ix: Name of sensitive index
        target: Values of classification target
    Methods:
        get_sensitive: extract the sensitive value from a row
            or the sensitive column from a dataset
        
    """
    def __init__( self, dataset, sensitive=None, target=None):
        self.name = dataset

        # Warfarin dosage dataset
        if (dataset == 'iwpc'):
            self.num_data = pd.DataFrame.from_records(
                arff.load('data/iwpc/iwpc_train_class.arff'),
                columns=[
                    'index', 'race=black', 'race=asian', 'age', 'height', 'weight', 'amiodarone',
                    'cyp2c9=13', 'cyp2c9=12', 'cyp2c9=23', 'cyp2c9=33', 'cyp2c9=22',
                    'vkorc1=CT', 'vkorc1=TT', 'decr', 'dose'
                    ])
            self.sup_ind = {}
            self.sup_ind['race'] = ['race=black','race=asian']
            self.sup_ind['age'] = ['age']
            self.sup_ind['height'] = ['height']
            self.sup_ind['weight'] = ['weight']
            self.sup_ind['amiodarone'] = ['amiodarone']
            self.sup_ind['cyp2c9'] = ['cyp2c9=13','cyp2c9=12','cyp2c9=23','cyp2c9=33','cyp2c9=22']
            self.sup_ind['vkorc1'] = ['vkorc1=CT','vkorc1=TT']
            self.sup_ind['decr'] = ['decr']
            self.sup_ind['dose'] = ['dose']
            self.target_ix = 'dose'
            self.sensitive_ix = 'race=black'
            if sensitive is None:
                self.get_sensitive = (lambda X: X['race=black'])

            self.target = self.num_data['dose']
            self.num_data = self.num_data.drop(['index'], axis = 1)
            self.num_data = self.num_data.drop(self.sup_ind[self.target_ix], axis = 1)
            del self.sup_ind['dose']


        #Adult dataset
        elif (dataset == 'adult'):
            self.original_data = pd.read_csv(
                "data/adult/adult.data",
                names=[
                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                    "Occupation", "Relationship", "Race", "Gender", "Capital Gain", "Capital Loss",
                    "Hours per week", "Country", "Target"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")
            del self.original_data['fnlwgt']
            self.sup_ind = make_super_indices(self.original_data)
            self.num_data = pd.get_dummies(self.original_data)
            self.target_ix = 'Target'
            self.sensitive_ix = sensitive

            #Define and dedup Target
            self.target = self.num_data['Target_>50K']
            self.num_data = self.num_data.drop(self.sup_ind[self.target_ix], axis = 1)
            del self.sup_ind['Target']

            #Dedup Gender
            self.num_data['Gender'] = self.num_data['Gender_Male']
            self.num_data = self.num_data.drop(self.sup_ind['Gender'], axis = 1)
            self.sup_ind['Gender'] = ['Gender']

            if sensitive is None:
                self.get_sensitive = (lambda X: X['Gender'])
            elif (sensitive == ''):
                self.get_sensitive = (lambda X: None)
            else:
                raise ValueError('Cannot handle sensitive '+sensitive+' in dataset '+dataset)


        #National Longitudinal Survey of Youth 97
        elif (dataset == 'nlsy97'):
            self.original_data = pd.read_csv(
                "data/nlsy97/20151026/processed_output.csv",
                names = ["PUBID.1997", "Gender", "Birth Year", "Census Region",
                    "Race", "Arrests", "Drug History", "Smoking History"],
                sep=r'\s*,\s*',
                engine='python',
                quoting=2,
                na_values="?")
            del self.original_data['PUBID.1997']
            self.target_ix = 'Arrests'
            self.sensitive_ix = sensitive
            self.sup_ind = make_super_indices(self.original_data)
            self.num_data = pd.get_dummies(self.original_data)

            #Define and dedup Target
            self.target = (self.num_data['Arrests'] > 0)*1.
            self.num_data = self.num_data.drop(self.sup_ind[self.target_ix], axis = 1)
            del self.sup_ind[self.target_ix]

            #Dedup Gender
            self.num_data['Gender'] = self.num_data['Gender_"Male"']
            self.num_data = self.num_data.drop(self.sup_ind['Gender'], axis = 1)
            self.sup_ind['Gender'] = ['Gender']

            if sensitive is None:
                self.get_sensitive = (lambda X: X['Gender'])
            elif (sensitive == 'Race'):
                self.get_sensitive = (lambda X: X['Race_"Black"'])
            else:
                raise ValueError('Cannot handle sensitive '+sensitive+' in dataset '+dataset)


        #German Datset (Incomplete)
        elif (dataset == 'german'):
        #http://programming-r-pro-bro.blogspot.com/2011/09/modelling-with-r-part-1.html
            original_data = pd.read_csv(
                "data/german/processed_output.csv",
                names = ["PUBID.1997", "Gender", "Birth Year", "Census Region",
                    "Race", "Arrests", "Drug History", "Smoking History"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")

        elif exists(dataset):
            print "loading new dataset %s" % dataset
            
            self.original_data = pd.read_csv(dataset)

            if target is None:
                target = self.original_data.columns[-1]
            self.target_ix = target
            if self.target_ix not in self.original_data:
                raise ValueError("unknown target feature %s" % self.target_ix)

            if sensitive is None:
                sensitive = self.original_data.columns[0]
            self.sensitive_ix = sensitive
            if self.sensitive_ix not in self.original_data:
                raise ValueError("unkown sensitive feature %s" % self.sensitive_ix)

            if self.sensitive_ix == self.target_ix:
                print "WARNING: target and sensitive attributes are the same (%s), I'm unsure whether this tool handles this case correctly" % target
            
            nominal_cols = set(self.original_data.select_dtypes(include=['object']).columns)
            
            self.num_data = pd.get_dummies(
                self.original_data,
                prefix_sep='_',
                columns=nominal_cols-set([target,sensitive]))

            self.num_data = self.num_data.apply(encode_nominal)
            
            self.sup_ind = make_super_indices(self.original_data)

            if self.target_ix in nominal_cols:
                targets = len(set(self.original_data[target]))
                if targets > 2:
                    print "WARNING: target feature %s has more than 2 values (it has %d), I'm unsure whether this tool handles that correctly" % (target, targets)
            del self.sup_ind[self.target_ix]
                #    self.target_ix = "%s_%s" % (self.target_ix,self.original_data[self.target_ix][0])

            if self.sensitive_ix in nominal_cols:
                targets = len(set(self.original_data[sensitive]))
                if targets > 2:
                    print "WARNING: sensitive feature %s has more than 2 values (it has %d), I'm unsure whether this tool handles that correctly" % (sensitive, targets)
                self.sup_ind[self.sensitive_ix] = [self.sensitive_ix]
                #    self.sensitive_ix = "%s_%s" % (self.sensitive_ix,self.original_data[self.sensitive_ix][0])

            self.target   = self.num_data[self.target_ix]
            self.num_data = self.num_data.drop([self.target_ix], axis = 1)
            
            self.get_sensitive = lambda X: X[self.sensitive_ix]

            print "target feature    = %s" % self.target_ix
            print "sensitive feature = %s" % self.sensitive_ix

        else:
            raise ValueError("Unknown dataset %s" % dataset)

    def delete_index ( self, index ):
        self.num_data.drop(self.sup_ind[index], axis = 1)
        del self.sup_ind[index]


#Categorical features are encoded as binary features, one for each category
#A super index keeps track of the mapping between a feature and its binary representation
def make_super_indices( dataset ):
    sup_ind = {}
    for i in dataset.columns:
        if dataset[i].dtype != 'O':
            sup_ind[i] = [i]
        else:
            unique = filter(lambda v: v==v, dataset[i].unique())
            sup_ind[i] = [i + '_' + s for s in unique]
    return sup_ind



## Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Name of dataset used')
    parser.add_argument('-m', '--measure',
                        default='average-unary-individual',
                        help='Quantity of interest',
                        choices=['average-unary-individual','unary-individual',
                                 'discrim', 'general-inf',
                                 'banzhaf','shapley'])
    parser.add_argument('-s', '--sensitive', default=None,     help='Sensitive field')
    parser.add_argument('-t', '--target',    default=None,     help='Target field', type=str)
    
    parser.add_argument('-e', '--erase-sensitive', action='store_false', help='Erase sensitive field from dataset')
    parser.add_argument('-p', '--show-plot', action='store_true', help='Output plot as pdf')
    parser.add_argument('-o', '--output-pdf', action='store_true', help='Output plot as pdf')
    parser.add_argument('-c', '--classifier', default='logistic', help='Classifier to use',
            choices=['logistic', 'svm', 'decision-tree', 'decision-forest'])

    parser.add_argument('--max_depth', default=2, help='Max depth for decision trees and forests')
    parser.add_argument('--n_estimators', default=20, help='Number of trees for decision forests')
    parser.add_argument('--seed', default=0, help='Random seed')
    
    parser.add_argument('-i', '--individual', default=0, type=int, help='Index for Individualized Transparency Report')
    parser.add_argument('-r', '--record-counterfactuals', action='store_true', help='Store counterfactual pairs for causal analysis')
    parser.add_argument('-a', '--active-iterations', type=int, default=10, help='Active Learning Iterations')
    return parser.parse_args()

def split_and_train_classifier(args, dataset, scaler=None):
    classifier = args.classifier
    ## Split data into training and test data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset.num_data, dataset.target, train_size=0.40)

    sens_train = dataset.get_sensitive(X_train)
    sens_test  = dataset.get_sensitive(X_test)

    if (scaler == None):
        #Initialize scaler to normalize training data
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)

    #Normalize all training and test data
    X_train = pd.DataFrame(scaler.transform(X_train), columns=(dataset.num_data.columns))
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=(dataset.num_data.columns))

    cls = train_classifier(args, X_train, y_train)
    
    return (cls, scaler, X_train, X_test, y_train, y_test, sens_train, sens_test)


def train_classifier(args, X_train, y_train):
    classifier = args.classifier
    #Initialize sklearn classifier model
    if (classifier == 'logistic'):
        import sklearn.linear_model as linear_model
        cls = linear_model.LogisticRegression(random_state=args.seed)
    elif (classifier == 'svm'):
        from sklearn import svm
        cls = svm.SVC(kernel='linear', cache_size=7000, random_state=args.seed)
    elif (classifier == 'decision-tree'):
        import sklearn.linear_model as linear_model
        cls = tree.DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.seed)
    elif (classifier == 'decision-forest'):
        from sklearn.ensemble import GradientBoostingClassifier
        cls = GradientBoostingClassifier(n_estimators=args.n_estimators,
                                         learning_rate=1.0,
                                         max_depth=args.max_depth,
                                         random_state=args.seed)

    #Train sklearn model
    cls.fit(X_train, y_train)
    return cls



def plot_series(series, args, xlabel, ylabel):
    plt.figure(figsize=(5,4))
    series.sort_values(inplace=True, ascending=False)
    #average_local_inf_series.plot(kind="bar", facecolor='#ff9999', edgecolor='white')
    series.plot(kind="bar")
    plt.xticks(rotation = 45, ha = 'right', size='small')
    plt.xlabel(xlabel, labelfont)
    plt.ylabel(ylabel, labelfont)
    plt.tight_layout()
    if (args.output_pdf == True):
        pp = PdfPages('figure-' + measure + '-' + args.dataset + '-' + args.classifier +'.pdf')
        print ('Writing to figure-' + measure + '-' + args.dataset + '-' + args.classifier + '.pdf')
        pp.savefig(bbox_inches='tight')
        pp.close()
    plt.show()


def plot_series_with_baseline(series, args, xlabel, ylabel, baseline):
    series.sort(ascending = True)
    plt.figure(figsize=(5,4))
    #plt.bar(range(series.size), series.as_matrix() - baseline)
    #(series - baseline).plot(kind="bar", facecolor='#ff9999', edgecolor='white')
    (series - baseline).plot(kind="bar")
    #plt.xticks(range(series.size), series.keys(), size='small')
    x1,x2,y1,y2 = plt.axis()
    X = range(series.size)
    for x,y in zip(X,series.as_matrix() - baseline):
        x_wd = 1. / series.size
        if(y < 0):
            plt.text(x+x_wd/2, y-0.01, '%.2f' % (y), ha='center', va= 'bottom', size='small')
        else:
            plt.text(x+x_wd/2, y+0.01, '%.2f' % (y), ha='center', va= 'top', size='small')
    plt.axis((x1,x2,-baseline,y2 + 0.01))
    plt.xticks(rotation = 45, ha = 'right', size='small')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: '%1.2f' % (x + baseline)))
    plt.axhline(linestyle = 'dashed', color = 'black')
    plt.text(x_wd, 0, 'Original Discrimination', ha = 'left', va = 'bottom')
    plt.xlabel(xlabel, labelfont)
    plt.ylabel(ylabel, labelfont)
    plt.tight_layout()
    if (args.output_pdf == True):
        pp = PdfPages('figure-' + measure + '-' + dataset.name + '-' + dataset.sensitive_ix + '-' + args.classifier + '.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + dataset.sensitive_ix + '-' + args.classifier + '.pdf')
        pp.savefig()
        pp.close()
    plt.show()


def measure_analytics(dataset, cls, X, y, sens=None):
    y_pred = cls.predict(X)

    error_rate = numpy.mean((y_pred != y)*1.)
    print('test error rate: %.3f' % error_rate)

    discrim0 =  qii.discrim(numpy.array(X), cls, numpy.array(sens))
    print('Initial Discrimination: %.3f' % discrim0)

    from scipy.stats.stats import pearsonr
    corr0 = pearsonr(sens, y)[0]
    print('Correlation: %.3f' % corr0)

    ji =  metrics.jaccard_similarity_score(y, sens)
    print('JI: %.3f' % ji)

    mi = metrics.normalized_mutual_info_score(y, sens)
    print('MI: %.3f' % mi)

