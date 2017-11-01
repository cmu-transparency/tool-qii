""" Various QII related computations. """

import pandas as pd
import numpy

RECORD_COUNTERFACTUALS = False

def intervene(X, features, x0):
    """ Constant intervention """

    X = numpy.array(X, copy=True)
    x0 = x0.T
    for f in features:
        X[:, f] = x0[f]
    return X

def causal_measure(clf, X, ep_state, f, x0):
    """ Causal Measure with a constant intervention. """

    c0 = clf.predict(x0)
    X1 = intervene(X, ep_state, x0)
    p1 = numpy.mean(1.*(clf.predict(X1) == c0))

    X2 = intervene(X, ep_state + [f], x0)
    p2 = numpy.mean(1.*(clf.predict(X2) == c0))

    return p2 - p1

def random_intervene(X, cols):
    """ Randomly intervene on a a set of columns of X. """

    n = X.shape[0]
    order = numpy.random.permutation(range(n))
    X_int = numpy.array(X)
    for c in cols:
        X_int[:, c] = X_int[order, c]
    return X_int

def random_intervene_class(X, target_class_X, cols):
    """ Randomly intervene on a a set of columns of target_class_X. """
    n = X.shape[0]
    p = target_class_X.shape[0]
    order = numpy.random.choice(n,p)
    target_int = numpy.array(target_class_X)
    for c in cols:
        target_int[:, c] = X[order, c]
    return target_int


def random_intervene_point(X, cols, x0):
    """ Randomly intervene on a a set of columns of x from X. """
    n = X.shape[0]
    order = numpy.random.permutation(range(n))
    X_int = numpy.tile(x0, (n, 1))
    for c in cols:
        X_int[:, c] = X[order, c]
    return X_int

def discrim(X, cls, sens):
    not_sens = 1 - sens
    y_pred = cls.predict(X)
    discrim = numpy.abs(numpy.dot(y_pred, not_sens)/sum(not_sens)
                        - numpy.dot(y_pred, sens)/sum(sens))
    return discrim

def discrim_ratio(X, cls, sens):
    not_sens = 1 - sens
    y_pred = cls.predict(X)
    sens_rate = numpy.dot(y_pred, sens)/sum(sens)
    not_sens_rate = numpy.dot(y_pred, not_sens)/sum(not_sens)

    discrim = not_sens_rate/sens_rate
    return discrim

def discrim_influence(dataset, cls, X_test, sens_test):
    """ Measure influence on discrimination. """

    discrim_inf = {}
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    for sf in sup_ind:
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        X_inter = random_intervene(numpy.array(X_test), ls)
        discrim_inter = discrim(X_inter, cls, numpy.array(sens_test))
        discrim_inf[sf] = discrim_inter
        print 'Discrimination %s: %.3f' % (sf, discrim_inf[sf])
    return discrim_inf

def average_local_influence(dataset, cls, X):
    average_local_inf = {}
    counterfactuals = {}
    iters = 10
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    y_pred = cls.predict(X)
    for sf in sup_ind:
        local_influence = numpy.zeros(y_pred.shape[0])
        if RECORD_COUNTERFACTUALS:
            counterfactuals[sf] = (numpy.tile(X, (iters, 1)), numpy.tile(X, (iters, 1)))
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for i in xrange(0, iters):
            X_inter = random_intervene(numpy.array(X), ls)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.
            if RECORD_COUNTERFACTUALS:
                n = X_inter.shape[0]
                counterfactuals[sf][1][i*n:(i+1)*n] = X_inter

        average_local_inf[sf] = 1 - (local_influence/iters).mean()
        #print('Influence %s: %.3f' % (sf, average_local_inf[sf]))
    return (average_local_inf, counterfactuals)

def average_local_class_influence(dataset, cls, X, target_class_X):
    average_local_inf_class = {}
    counterfactuals = {}
    iters = 10
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    y_pred = cls.predict(target_class_X)
    for sf in sup_ind:
        local_influence = numpy.zeros(y_pred.shape[0])
        if RECORD_COUNTERFACTUALS:
            counterfactuals[sf] = (numpy.tile(target_class_X, (iters, 1)), numpy.tile(target_class_X, (iters, 1)))
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for i in xrange(0, iters):
            X_inter = random_intervene_class(numpy.array(X), numpy.array(target_class_X), ls)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.
            if RECORD_COUNTERFACTUALS:
                n = X_inter.shape[0]
                counterfactuals[sf][1][i*n:(i+1)*n] = X_inter

        average_local_inf_class[sf] = 1 - (local_influence/iters).mean()
        #print('Influence %s: %.3f' % (sf, average_local_inf_class[sf]))
    return (average_local_inf_class, counterfactuals)


def unary_individual_influence(dataset, cls, x_ind, X):
    y_pred = cls.predict(x_ind.reshape(1, -1))
    average_local_inf = {}
    counterfactuals = {}
    iters = 1
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    for sf in sup_ind:
        local_influence = numpy.zeros(y_pred.shape[0])
        if RECORD_COUNTERFACTUALS:
            counterfactuals[sf] = (numpy.tile(X, (iters, 1)), numpy.tile(X, (iters, 1)))
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        for i in xrange(0, iters):
            X_inter = random_intervene_point(numpy.array(X), ls, x_ind)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.
            if RECORD_COUNTERFACTUALS:
                n = X_inter.shape[0]
                counterfactuals[sf][1][i*n:(i+1)*n] = X_inter

        average_local_inf[sf] = 1 - (local_influence/iters).mean()
        #print('Influence %s: %.3f' % (sf, average_local_inf[sf]))
    return (average_local_inf, counterfactuals)

def shapley_influence(dataset, cls, x_individual, X_test):
    p_samples = 600
    s_samples = 600

    def v(S, x, X_inter):
        x_rep = numpy.tile(x, (p_samples, 1))
        for f in S:
            x_rep[:, f] = X_inter[:, f]
        p = ((cls.predict(x_rep) == y0)*1.).mean()
        return (p, x_rep)

    #min_i = numpy.argmin(sum_local_influence)
    y0 = cls.predict(x_individual)
    print y0
    b = numpy.random.randint(0, X_test.shape[0], p_samples)
    X_sample = numpy.array(X_test.ix[b])
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    super_indices = dataset.sup_ind.keys()

    shapley = dict.fromkeys(super_indices, 0)
    if RECORD_COUNTERFACTUALS:
        base = numpy.tile(x_individual, (2*p_samples*s_samples, 1))
        #counterfactuals = dict([(sf, (base, numpy.zeros(p_samples*s_samples*2, X_test.shape[1])))
        #    for sf in dataset.sup_ind.keys()])

        counterfactuals = dict([(sf, (base,
                                      numpy.zeros((p_samples*s_samples*2, X_test.shape[1]))))
                                for sf in dataset.sup_ind.keys()])
    else:
        counterfactuals = {}

    for sample in xrange(0, s_samples):
        perm = numpy.random.permutation(len(super_indices))
        for i in xrange(0, len(super_indices)):
            # Choose a random subset and get string indices by flattening
            #  excluding si
            si = super_indices[perm[i]]
            S_m_si = sum([sup_ind[super_indices[perm[j]]] for j in xrange(0, i)], [])
            #translate into intiger indices
            ls_m_si = [f_columns.get_loc(f) for f in S_m_si]
            #repeat x_individual_rep
            (p_S, X_S) = v(ls_m_si, x_individual, X_sample)
            #also intervene on s_i
            ls_si = [f_columns.get_loc(f) for f in sup_ind[si]]
            (p_S_si, X_S_si) = v(ls_m_si + ls_si, x_individual, X_sample)
            shapley[si] = shapley[si] - (p_S_si - p_S)/s_samples

            if RECORD_COUNTERFACTUALS:
                start_ind = 2*sample*p_samples
                mid_ind = (2*sample+1)*p_samples
                end_ind = 2*(sample+1)*p_samples
                counterfactuals[si][1][start_ind:mid_ind] = X_S
                counterfactuals[si][1][mid_ind:end_ind] = X_S_si

    return (shapley, counterfactuals)

def banzhaf_influence(dataset, cls, x_individual, X_test):
    p_samples = 600
    s_samples = 600

    def v(S, x, X_inter):
        x_rep = numpy.tile(x, (p_samples, 1))
        for f in S:
            x_rep[:, f] = X_inter[:, f]
        p = ((cls.predict(x_rep) == y0)*1.).mean()
        return p

    #min_i = numpy.argmin(sum_local_influence)
    y0 = cls.predict(x_individual)
    b = numpy.random.randint(0, X_test.shape[0], p_samples)
    X_sample = numpy.array(X_test.ix[b])
    f_columns = dataset.num_data.columns
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
            #repeat x_individual_rep
            p_S = v(ls_m_si, x_individual, X_sample)
            #also intervene on s_i
            ls_si = [f_columns.get_loc(f) for f in sup_ind[si]]
            p_S_si = v(ls_m_si + ls_si, x_individual, X_sample)
            banzhaf[si] = banzhaf[si] - (p_S - p_S_si)/s_samples
    return banzhaf

def analyze_outliers(counterfactuals, out_cls, cls):
    outlier_fracs = {}
    new_outlier_fracs = {}
    qii = {}
    for sf, pairs in counterfactuals.iteritems():
        X = pairs[0]
        X_cf = pairs[1]
        outs_X = out_cls.predict(X) == -1
        outs_X_cf = out_cls.predict(X_cf) == -1
        outlier_fracs[sf] = numpy.mean(outs_X_cf)
        lnot = numpy.logical_not
        land = numpy.logical_and
        old_outlier_frac = numpy.mean(lnot(outs_X))
        new_outlier_fracs[sf] = numpy.mean(land(lnot(outs_X), outs_X_cf))/old_outlier_frac
        qii = numpy.mean(cls.predict(X) != cls.predict(X_cf))
        print 'QII %s %.3f' % (sf, qii)
    return (outlier_fracs, new_outlier_fracs)
