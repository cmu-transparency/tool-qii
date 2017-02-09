# QII code from book chapter submission for archival purposes
import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
from sklearn.metrics import jaccard_similarity_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy
import numpy.random

import numpy.linalg
import sys
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import time


#labelfont = {'fontname':'Times New Roman', 'size':15}
labelfont = {}
plt.style.use('bmh')
figsize = (6,4)
#hfont = {'fontname':'Helvetica'}

def dp_noise(eps, sens):
    return numpy.random.laplace(scale = sens/eps)



#Constant intervention
def intervene( X, features, x0 ):
    X = numpy.array(X, copy=True)
    x0 = x0.T
    for f in features:
        X[:,f] = x0[f]
    return X

#Causal Measure with a constant intervention
def causal_measure ( clf, X, ep_state, f, x0 ):
    c0 = clf.predict(x0)
    X1 = intervene( X, ep_state, x0 )
    p1 = numpy.mean(1.*(clf.predict(X1) == c0))

    X2 = intervene( X, ep_state + [f], x0 )
    p2 = numpy.mean(1.*(clf.predict(X2) == c0))

    return p2 - p1

#Randomly intervene on a a set of columns of X
def random_intervene( X, cols ):
    n = X.shape[0]
    order = numpy.random.permutation(range(n))
    X_int = numpy.array(X)
    for c in cols:
        X_int[:, c] = X_int[order, c]
    return X_int

def discrim (X, cls, sens):
    not_sens = 1 - sens
    y_pred = cls.predict(X)
    discrim = numpy.abs(numpy.dot(y_pred,not_sens)/sum(not_sens)
                         - numpy.dot(y_pred,sens)/sum(sens))
    return discrim

def discrim_ratio (X, cls, sens):
    not_sens = 1 - sens
    y_pred = cls.predict(X)
    sens_rate = numpy.dot(y_pred,sens)/sum(sens)
    not_sens_rate = numpy.dot(y_pred,not_sens)/sum(not_sens)

    discrim = not_sens_rate/sens_rate
    return discrim





from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
#X = preprocessing.scale(X)
#med = numpy.median(X[:,9])

class Dataset(object):
    """
    Attributes:
        name: A string representing the name of the dataset
        original_data: Dataset with categorical variables
        num_data: Dataset with cleaned numeric values
        sup_ind: Super Index containing a dict which maps from original
            feature to list of dummy features
        target_ix: Name of target index
        sensitive_ix: Name of sensitive index
        target: Values of classification target
    """
    def __init__( self, dataset, sensitive ):
        self.name = dataset


        if (dataset == 'adult'):
            self.original_data = pd.read_csv(
                "data/adult/adult.data",
                names=[
                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
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

            #Dedup Sex
            self.num_data['Sex'] = self.num_data['Sex_Male']
            self.num_data = self.num_data.drop(self.sup_ind['Sex'], axis = 1)
            self.sup_ind['Sex'] = ['Sex']

            if (sensitive == 'Sex'):
                self.sensitive = (lambda X: X['Sex'])
            else:
                raise ValueError('Cannot handle sensitive '+sensitive+' in dataset '+dataset)


        if (dataset == 'nlsy97'):
            self.original_data = pd.read_csv(
                "data/nlsy97/20151026/processed_output.csv",
                names = ["PUBID.1997", "Sex", "Birth Year", "Census Region",
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

            #Dedup Sex
            self.num_data['Sex'] = self.num_data['Sex_"Male"']
            self.num_data = self.num_data.drop(self.sup_ind['Sex'], axis = 1)
            self.sup_ind['Sex'] = ['Sex']

            if (sensitive == 'Sex'):
                self.sensitive = (lambda X: X['Sex'])
            elif (sensitive == 'Race'):
                self.sensitive = (lambda X: X['Race_"Black"'])
            else:
                raise ValueError('Cannot handle sensitive '+sensitive+' in dataset '+dataset)



        if (dataset == 'german'):
        #http://programming-r-pro-bro.blogspot.com/2011/09/modelling-with-r-part-1.html
            original_data = pd.read_csv(
                "data/german/processed_output.csv",
                names = ["PUBID.1997", "Sex", "Birth Year", "Census Region",
                    "Race", "Arrests", "Drug History", "Smoking History"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")

    def delete_index ( self, index ):
        self.num_data.drop(self.sup_ind[index], axis = 1)
        del self.sup_ind[index]


def make_super_indices( dataset ):
    sup_ind = {}
    for i in dataset.columns:
        if dataset[i].dtype != 'O':
            sup_ind[i] = [i]
        else:
            unique = filter(lambda v: v==v, dataset[i].unique())
            sup_ind[i] = [i + '_' + s for s in unique]
    return sup_ind

class ClsDiscrimWrapper(object):
    def __init__(self, cls, sens, n):
        self.cls = cls
        self.sens = sens
        self.rand = numpy.random.ranf(n)

    def predict(self, X, p):
        s = (self.sens(X) > 0) * 1.
        r = self.rand  < p
        mask =  1. - (s * r)
        y = self.cls.predict(X)
        return 1. - (1. - y) * mask


def discrim_influence(dataset, cls, X_test, sens_test):
    discrim_inf = {}
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    for sf in sup_ind:
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        X_inter = random_intervene(numpy.array(X_test), ls)
        discrim_inter = discrim(X_inter, cls, numpy.array(sens_test))
        discrim_inf[sf] = discrim_inter
        print('Discrimination %s: %.3f' % (sf, discrim_inf[sf]))
    return discrim_inf


#def main():

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Name of dataset used')
parser.add_argument('-m', '--measure', default='average-local-inf', help='Quantity of interest')
parser.add_argument('-s', '--sensitive', default='Sex', help='Sensitive field')
parser.add_argument('-e', '--erase-sensitive', action='store_false', help='Erase sensitive field from dataset')
parser.add_argument('-p', '--output-pdf', action='store_true', help='Output plot as pdf')
parser.add_argument('-c', '--classifier', default='logistic', help='Classifier to use',
        choices=['logistic', 'svm', 'decision-tree', 'decision-forest'])
args = parser.parse_args()

dataset = Dataset(args.dataset, args.sensitive)
#if (args.erase_sensitive):
#  print 'Erasing sensitive'
#  dataset.delete_index(args.sensitive)

measure = args.measure

f_columns = dataset.num_data.columns
sup_ind = dataset.sup_ind

## Train Classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset.num_data, dataset.target, train_size=0.40)


sens_train = dataset.sensitive(X_train)
sens_test  = dataset.sensitive(X_test)

scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=(dataset.num_data.columns))
X_test = pd.DataFrame(scaler.transform(X_test), columns=(dataset.num_data.columns))

if (args.classifier == 'logistic'):
    import sklearn.linear_model as linear_model
    cls = linear_model.LogisticRegression()
elif (args.classifier == 'svm'):
    from sklearn import svm
    cls = svm.SVC(kernel='linear', cache_size=7000)
elif (args.classifier == 'decision-tree'):
    import sklearn.linear_model as linear_model
    cls = tree.DecisionTreeClassifier()
elif (args.classifier == 'decision-forest'):
    from sklearn.ensemble import GradientBoostingClassifier
    cls = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=2, random_state=0)

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

def entropy(p):
    p = (p + 0.00000001)/1.000002
    return p*numpy.log(p) + (1-p)*numpy.log(1-p)


error_rate = numpy.mean((y_pred != y_test)*1.)
print('error rate: %.3f' % error_rate)



discrim0 =  discrim(numpy.array(X_test), cls, numpy.array(sens_test))
print('Initial Discrimination: %.3f' % discrim0)


from scipy.stats.stats import pearsonr
corr0 = pearsonr(sens_test, y_test)[0]
print('Correlation: %.3f' % corr0)

ji =  metrics.jaccard_similarity_score(y_test, sens_test)
print('JI: %.3f' % ji)

mi = metrics.normalized_mutual_info_score(y_test, sens_test)
print('MI: %.3f' % mi)

t0 = time.time()
if measure == 'discrim-inf':
    baseline = discrim0
    discrim_inf = discrim_influence(dataset, cls, X_test, sens_test)
    discrim_inf_series = pd.Series(discrim_inf, index = discrim_inf.keys())
    discrim_inf_series.sort(ascending = True)
    plt.figure(figsize=figsize)
    #plt.bar(range(discrim_inf_series.size), discrim_inf_series.as_matrix() - baseline)
    #(discrim_inf_series - baseline).plot(kind="bar", facecolor='#ff9999', edgecolor='white')
    (discrim_inf_series - baseline).plot(kind="bar")
    #plt.xticks(range(discrim_inf_series.size), discrim_inf_series.keys(), size='small')
    x1,x2,y1,y2 = plt.axis()
    X = range(discrim_inf_series.size)
    #font = {'family' : 'normal',
    #        'weight' : 'bold',
    #        'size'   : 22}
    #matplotlib.rc('font', **font)
    for x,y in zip(X,discrim_inf_series.as_matrix() - baseline):
        x_wd = 1. / discrim_inf_series.size
        if(y < 0):
            plt.text(x+x_wd/2, y-0.015, '%.2f' % (y), ha='center', va= 'bottom', size='small')
        else:
            plt.text(x+x_wd/2, y+0.015, '%.2f' % (y), ha='center', va= 'top', size='small')
    plt.axis((x1,x2,-baseline,y2 + 0.01))
    plt.xticks(rotation = 45, ha = 'right', size=13)
    plt.yticks(size=13)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: '%1.2f' % (x + baseline)))
    plt.axhline(linestyle = 'dashed', color = 'black')
    plt.text(x_wd, 0, 'Original Disparity', ha = 'left', va = 'bottom')
    plt.xlabel('Feature', labelfont)
    plt.ylabel('QII on Group Disparity', labelfont)
    plt.tight_layout()
    if (args.output_pdf == True):
        pp = PdfPages('figures-bookchapter/figure-' + measure + '-' + dataset.name + '-' + dataset.sensitive_ix + '-' + args.classifier + '.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + dataset.sensitive_ix + '-' + args.classifier + '.pdf')
        pp.savefig()
        pp.close()
    plt.show()

if measure == 'average-local-inf':
    average_local_inf = {}
    iters = 2
    y_pred = cls.predict(X_test)
    for sf in dataset.sup_ind:
        local_influence = numpy.zeros(y_pred.shape[0])
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for i in xrange(0, iters):
            X_inter = random_intervene(numpy.array(X_test), ls)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.

        average_local_inf[sf] = 1 - (local_influence/iters).mean()
        print('Average Local Influence %s: %.3f' % (sf, average_local_inf[sf]))

    plt.figure(figsize=figsize)
    average_local_inf_series = pd.Series(average_local_inf, index = average_local_inf.keys())
    average_local_inf_series.sort(ascending = False)
    #average_local_inf_series.plot(kind="bar", facecolor='#ff9999', edgecolor='white')
    average_local_inf_series.plot(kind="bar")
    plt.xticks(rotation = 45, ha = 'right', size=13)
    plt.yticks(size=13)
    plt.xlabel('Feature', labelfont)
    plt.ylabel('QII on Outcomes', labelfont)
    plt.tight_layout()
    #from matplotlib import rcParams
    #rcParams.update({'figure.autolayout': True})
    #if (args.classifier == 'decision-tree' or args.classifier == 'decision-forest'):
    #    fi = pd.Series(cls.feature_importances_, index=dataset.num_data.columns)
    #    sfi = [fi.loc[dataset.sup_ind[sf]].sum() for sf in dataset.sup_ind]
    #   super_indices = [sf for sf in dataset.sup_ind]
    #   sfi = pd.Series(sfi, index = super_indices)
    #   plt.figure(figsize=(5,4))
    #   sfi.sort(ascending = False)
    #   sfi.plot(kind='bar')
    if (args.output_pdf == True):
        pp = PdfPages('figures-bookchapter/figure-' + measure + '-' + dataset.name + '-' + args.classifier +'.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        pp.savefig(bbox_inches='tight')
        pp.close()
    plt.show()


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

    plt.figure(figsize=figsize)
    average_local_inf_series = pd.Series(average_local_inf, index = average_local_inf.keys())
    average_local_inf_series.sort(ascending = False)
    average_local_inf_series.plot(kind="bar")
    if (args.output_pdf == True):
        pp = PdfPages('figures-bookchapter/figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        pp.savefig()
        pp.close()
    plt.show()

if measure == 'sensitivity':
    average_local_inf = {}
    samples = 50
    iters = 10
    cls = ClsDiscrimWrapper(cls, dataset.sensitive, X_test.shape[0])

    ps = np.zeros(iters+1)
    infs_sens = np.zeros(iters+1)
    for i in xrange(0, iters + 1):
        sf = dataset.sensitive_ix
        p = 1./(iters)*i
        ps[i] = p
        local_influence = numpy.zeros(y_pred.shape[0])
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for s in xrange(0, samples):
            y_pred = cls.predict(X_test, p)
            X_inter = pd.DataFrame(random_intervene(numpy.array(X_test), ls), columns = X_test.columns)
            y_pred_inter = cls.predict(X_inter, p)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.

        infs_sens[i] = 1 - (local_influence/samples).mean()
        print('Average Local Influence %.3f: %.3f' % (p, infs_sens[i]))
    sf = 'Marital Status'
    if (dataset.name == 'adult'):
        sf = 'Marital Status'
    elif (dataset.name == 'nlsy97'):
        sf = 'Drug History'
    infs_marital = np.zeros(iters+1)
    for i in xrange(0, iters + 1):
        p = 1./(iters)*i
        ps[i] = p
        local_influence = numpy.zeros(y_pred.shape[0])
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for s in xrange(0, samples):
            y_pred = cls.predict(X_test, p)
            X_inter = pd.DataFrame(random_intervene(numpy.array(X_test), ls), columns = X_test.columns)
            y_pred_inter = cls.predict(X_inter, p)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.

        infs_marital[i] = 1 - (local_influence/samples).mean()
        print('Average Local Influence %.3f: %.3f' % (p, infs_marital[i]))

    plt.figure(figsize=figsize)
    sens_p = plt.plot(ps, infs_sens, linewidth = 2.0, label = dataset.sensitive_ix)
    sf_p = plt.plot(ps, infs_marital, linewidth = 2.0, label = sf)
    plt.xlabel('Fraction of Discriminatory Zip Codes', labelfont)
    plt.ylabel('QII on Outcomes', labelfont)
    plt.xticks(size=13)
    plt.yticks(size=13)
    #plt.legend(handles=[sens_p, sf_p])
    plt.legend(loc=2, fancybox=True)
    plt.tight_layout()
    if (args.output_pdf == True):
        pp = PdfPages('figures-bookchapter/figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        pp.savefig()
        pp.close()
    plt.show()

if measure == 'banzhaf':
   # local_infs = {}
   # iters = 100
   # y_pred = cls.predict(X_test)
   # max_local_influence = numpy.zeros(y_pred.shape[0])
   # for sf in dataset.sup_ind:
   #     local_influence = numpy.zeros(y_pred.shape[0])
   #     ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
   #     for i in xrange(0, iters):
   #         X_inter = random_intervene(numpy.array(X_test), ls)
   #         y_pred_inter = cls.predict(X_inter)
   #         local_influence = local_influence + (y_pred == y_pred_inter)*1.

   #     local_infs[sf] = 1 - local_influence/iters
   #     max_local_influence = numpy.maximum(max_local_influence, local_infs[sf])

   # #print ('Min: %.3f' % (sum_local_influence.min()))
   # #print ('Max: %.3f' % (sum_local_influence.max()))
   # #print ('Avg: %.3f' % (sum_local_influence.mean()))

   # hist, bins = numpy.histogram(max_local_influence, bins=20)
   # width = 0.7 * (bins[1] - bins[0])
   # center = (bins[:-1] + bins[1:]) / 2
   # #plt.yscale('log')
   # plt.ylabel('Number of individuals')
   # plt.xlabel('Maximum Influence of some input')
   # plt.bar(center, hist, align='center', width=width)
   # if (args.output_pdf == True):
   #     pp = PdfPages('figure-' + measure + '-' + dataset.name + '-' + dataset.sensitive_ix + '-' + args.classifier + '-influence-hist.pdf')
   #     print ('Writing to figure-' + measure + '-' + dataset.name + '-' + dataset.sensitive_ix + '-' + args.classifier + '.pdf')
   #     pp.savefig()
   #     pp.close()

   # plt.show()
   # return 0
    p_samples = 600
    s_samples = 600

    def v(S, x, X_inter):
        x_rep = numpy.tile(x, (p_samples, 1))
        for f in S:
            x_rep[:,f] = X_inter[:,f]
        p = ((cls.predict(x_rep) == y0)*1.).mean()
        return p

    #min_i = numpy.argmin(sum_local_influence)
    min_i = 0
    print min_i
    x_min = X_test.ix[min_i]
    y0 = cls.predict(x_min)
    b = np.random.randint(0,X_test.shape[0],p_samples)
    X_sample = numpy.array(X_test.ix[b])
    sup_ind = dataset.sup_ind
    super_indices = dataset.sup_ind.keys()

    banzhaf = dict.fromkeys(super_indices, 0)

    for sample in xrange(0, s_samples):
        r = numpy.random.ranf(len(super_indices))
        S = [super_indices[i] for i in xrange(0, len(super_indices)) if r[i] > 0.5]
        for si in super_indices:
            # Choose a random subset and get string indices by flattening
            #  excluding si
            S_m_si = sum([sup_ind[x] for x in S if x != si], [])
            #translate into intiger indices
            ls_m_si = [f_columns.get_loc(f) for f in S_m_si]
            #repeat x_min_rep
            p_S = v(ls_m_si, x_min, X_sample)
            #also intervene on s_i
            ls_si = [f_columns.get_loc(f) for f in sup_ind[si]]
            p_S_si = v(ls_m_si + ls_si, x_min, X_sample)
            banzhaf[si] = banzhaf[si] - (p_S - p_S_si)/s_samples

    banzhaf_series = pd.Series(banzhaf, index = banzhaf.keys())
    banzhaf_series.sort(ascending = False)
    banzhaf_series.plot(kind="bar", facecolor='#ff9999', edgecolor='white')

    if (args.output_pdf == True):
        pp = PdfPages('figures-bookchapter/figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        pp.savefig()
        pp.close()
    plt.show()



if measure == 'shapley':

    p_samples = 600
    s_samples = 600

    def v(S, x, X_inter):
        x_rep = numpy.tile(x, (p_samples, 1))
        for f in S:
            x_rep[:,f] = X_inter[:,f]
        p = ((cls.predict(x_rep) == y0)*1.).mean()
        return p


    #min_i = numpy.argmin(sum_local_influence)
    min_i = 0
    print min_i
    x_min = X_test.ix[min_i]
    y0 = cls.predict(x_min)
    b = np.random.randint(0,X_test.shape[0],p_samples)
    X_sample = numpy.array(X_test.ix[b])
    sup_ind = dataset.sup_ind
    super_indices = dataset.sup_ind.keys()

    shapley = dict.fromkeys(super_indices, 0)

    for sample in xrange(0, s_samples):
        perm = np.random.permutation(len(super_indices))
        for i in xrange(0, len(super_indices)):
            # Choose a random subset and get string indices by flattening
            #  excluding si
            si = super_indices[perm[i]]
            S_m_si = sum([sup_ind[super_indices[perm[j]]] for j in xrange(0, i)], [])
            #translate into intiger indices
            ls_m_si = [f_columns.get_loc(f) for f in S_m_si]
            #repeat x_min_rep
            p_S = v(ls_m_si, x_min, X_sample)
            #also intervene on s_i
            ls_si = [f_columns.get_loc(f) for f in sup_ind[si]]
            p_S_si = v(ls_m_si + ls_si, x_min, X_sample)
            shapley[si] = shapley[si] - (p_S_si - p_S)/s_samples


    plt.figure(figsize=figsize)
    shapley_series = pd.Series(shapley, index = shapley.keys())
    shapley_series.sort(ascending = False)
    shapley_series.plot(kind="bar", facecolor='#ff9999', edgecolor='white')
    plt.ylabel('QII on Outcomes (Shapley)')
    plt.xticks(rotation = 45, ha = 'right', size=13)
    plt.yticks(size=12)
    plt.tight_layout()

    if (args.output_pdf == True):
        pp = PdfPages('figures-bookchapter/figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        print ('Writing to figure-' + measure + '-' + dataset.name + '-' + args.classifier + '.pdf')
        pp.savefig()
        pp.close()
    plt.show()


t1 = time.time()
print (t1 - t0)







#if __name__ == '__main__':
#    main()


