import numpy as np

def shapley_influence(cls, x_individual, X_test, columns, column_groups=None):
    p_samples = 600
    s_samples = 600

    def v(S, x, X_inter):
        x_rep = np.tile(x, (p_samples, 1))
        for f in S:
            x_rep[:, f] = X_inter[:, f]
        #p = ((cls.predict(x_rep) == y0)*1.).mean()
        p = ((cls.predict(x_rep) == 1)*1.).mean()
        return (p, x_rep)

    #min_i = np.argmin(sum_local_influence)
    y0 = cls.predict(x_individual)
    print y0
    b = np.random.randint(0, X_test.shape[0], p_samples)
    X_sample = X_test[b, :]
    f_columns = columns
    if (column_groups != None):
        sup_ind = column_groups
    else:
        sup_ind = {column: [columns.index(column)] for column in columns}
        super_indices = sup_ind.keys()


    shapley = dict.fromkeys(super_indices, 0)

    for sample in xrange(0, s_samples):
        perm = np.random.permutation(len(super_indices))
        for i in xrange(0, len(super_indices)):
            # Choose a random subset and get string indices by flattening
            #  excluding si
            si = super_indices[perm[i]]
            S_m_si = sum([sup_ind[super_indices[perm[j]]] for j in xrange(0, i)], [])
            #translate into integer indices
            ls_m_si = S_m_si
            #repeat x_individual_rep
            (p_S, X_S) = v(ls_m_si, x_individual, X_sample)
            #also intervene on s_i
            ls_si = sup_ind[si]
            (p_S_si, X_S_si) = v(ls_m_si + ls_si, x_individual, X_sample)
            shapley[si] = shapley[si] - (p_S_si - p_S)/s_samples

    return shapley




def shapley_influence_score(cls, x_individual, X_test, columns, column_groups=None):
    p_samples = 600
    s_samples = 600
    import scipy.special as special

    def v(S, x, X_inter):
        x_rep = np.tile(x, (p_samples, 1))
        for f in S:
            x_rep[:, f] = X_inter[:, f]
        #p = ((cls.predict(x_rep) == y0)*1.).mean()
        p = special.logit(cls.predict_proba(x_rep))[:,0].mean()
        return (p, x_rep)

    #min_i = np.argmin(sum_local_influence)
    y0 = cls.predict(x_individual)
    print y0
    b = np.random.randint(0, X_test.shape[0], p_samples)
    X_sample = X_test[b, :]
    f_columns = columns
    if (column_groups != None):
        sup_ind = column_groups
    else:
        sup_ind = {column: [columns.index(column)] for column in columns}
        super_indices = sup_ind.keys()


    shapley = dict.fromkeys(super_indices, 0)

    for sample in xrange(0, s_samples):
        perm = np.random.permutation(len(super_indices))
        for i in xrange(0, len(super_indices)):
            # Choose a random subset and get string indices by flattening
            #  excluding si
            si = super_indices[perm[i]]
            S_m_si = sum([sup_ind[super_indices[perm[j]]] for j in xrange(0, i)], [])
            #translate into integer indices
            ls_m_si = S_m_si
            #repeat x_individual_rep
            (p_S, X_S) = v(ls_m_si, x_individual, X_sample)
            #also intervene on s_i
            ls_si = sup_ind[si]
            (p_S_si, X_S_si) = v(ls_m_si + ls_si, x_individual, X_sample)
            shapley[si] = shapley[si] - (p_S_si - p_S)/s_samples

    return shapley




def random_intervene_pop(X, X_pop, cols):
    """ Randomly intervene on a a set of columns of X. """

    m = X.shape[0]
    n = X_pop.shape[0]
    #order = np.random.permutation(range(n))
    order = np.random.randint(n, size=m)
    X_int = np.array(X)
    for c in cols:
        X_int[:, c] = X_pop[order, c]
    return X_int



def average_local_influence(cls, X, X_pop, columns, column_groups=None):
    average_local_inf = {}
    iters = 10
    f_columns = columns
    if (column_groups != None):
        sup_ind = column_groups
    else:
        sup_ind = {column: [columns.index(column)] for column in columns}
        super_indices = sup_ind.keys()

    y_pred = cls.predict(X)
    for sf in sup_ind:
        local_influence = np.zeros(y_pred.shape[0])
        ls = sup_ind[sf]
        for i in xrange(0, iters):
            X_inter = random_intervene_pop(X, X_pop, ls)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.

        average_local_inf[sf] = 1 - (local_influence/iters).mean()
        #print('Influence %s: %.3f' % (sf, average_local_inf[sf]))
    return average_local_inf

