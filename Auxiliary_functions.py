# This Code has been developed by Javier Olias.
from pandas import DataFrame as df
import numpy as np
from numpy import linalg as la
from pyriemann.utils.distance import distance_riemann as rd
from pyriemann.utils.mean import mean_riemann as rm
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.tangentspace import tangent_space as ts


def split_data(data: dict, ntest: int, ntr: int, seed=1):
    """
    Function that randomly  split the data into training and test groups.
    It was made for a specific data structure built in Matlab, modify it at convenience
    :param data: Data from the .mat file
    :param ntest: Number of test trials
    :param ntr: Number of Training Trials
    :param seed: Seed for replication purposes
    :return:
        tra: Training trials as DataFrame
        tes: Test trials as DataFrame
        tesy: Test trials classes
        ta: Training trials indices
        te: Test trials indices
    """

    n = data['DATA']['y'][0][0].shape[0]
    cc = np.unique(data['DATA']['y'][0][0])
    assert cc.shape.__len__() == 1
    nc = cc.shape[0]
    cl = np.stack(
        [np.where(x == data['DATA']['y'][0][0])[0] for x in cc])

    te = []
    ta = []
    np.random.seed(seed)
    for k in range(nc):
        oo = np.random.permutation(int(np.floor(n / nc)))
        te = te + [cl[k][oo[:ntest // nc]], ]
        ta = ta + [cl[k][oo[-ntr // nc:]], ]
    te = np.sort(np.stack(te, 1).reshape(-1))
    ta = np.sort(np.stack(ta, 1).reshape(-1))

    tra = df({
        'mindex': ta,
        'y': np.reshape(data['DATA']['y'][0][0][ta], ntr),
        'x': [item[0] for item in
              np.split(np.transpose(data['DATA']['x'][0][0][:, :, ta], (2, 0, 1)), ntr, axis=0)],
    })
    tes = df({
        'mindex': te,
        'x': [item[0] for item in
              np.split(np.transpose(data['DATA']['x'][0][0][:, :, te], (2, 0, 1)), ntest, axis=0)],
    })
    tesy = [item[0, 0] for item in np.split(data['DATA']['y'][0][0][te], range(1, ntest))]
    return tra, tes, tesy, ta, te


def csp_function(ca, p, ce=None, tr=False):
    """
    Compute CSP Filters.
    :param ca: Training trials Covariances as DataFrame
    :param p: Number of filters
    :param ce: Test trials Covariances as DataFrame
    :param tr: Trace normalization.
    :return: If no test trials are passed it returns the CSP filters.
        If test trials are passed it filters and returns them.
    """
    ca2 = ca.x.apply(lambda x: x / np.trace(x)) if tr else ca.x
    cc = np.unique(ca.y)
    if 2 < cc.size:
        raise Exception('CSP only works for two classes')
    if 0 != p % 2:
        raise Exception('please enter an even number of features')
    c1 = np.where(cc[0] == ca.y)[0]

    co1 = np.mean(np.stack(ca2[c1], 0), 0)
    co = np.mean(np.stack(ca2, 0), 0)
    val, vecs = la.eig(np.dot(la.inv(co), co1))
    vecs2 = vecs[:, np.argsort(val)]
    csp = vecs2[:, list(range(int(p/2)))+list(range(int(-p/2), 0))]

    # If test trials are passed, then it returns them filtered, otherwise it just returns the CSP filters.
    if ce is not None:
        trcovs = df({'x': ca.x.apply(lambda x: la.multi_dot([csp.T, x, csp]))})
        trcovs['y'] = ca.y
        tecovs = df({'x': ce.x.apply(lambda x: la.multi_dot([csp.T, x, csp]))})
    else:
        trcovs = csp
        tecovs = None
    return trcovs, tecovs


def lda_clasfier(ca, ce, opt, normalize=False):
    """
    LDA Classifier.
    :param ca: Filtered training trials Covariances as DataFrame
    :param ce: Filtered test trials Covariances as DataFrame
    :param opt: If 0 no shrinkage, if 1 LD shrinkage, if 2 Gaussian shrinkage, If 3 identity shrinkage
    :param normalize: Normalize features.
    :return: Label of test trials
    """
    cc = np.unique(ca.y)
    nc = cc.shape[0]

    features = np.stack(ca.x.apply(lambda x: np.diag(x)), 0)
    if normalize:
        features = features / features.sum(axis=1, keepdims=True)

    features = np.log(features)
    nt, ns = features.shape

    cl, ft, mm,  = [], [], []
    for k1 in range(nc):
        cl = cl + [np.where(cc[k1] == ca.y)[0], ]
        ft = ft + [features[cl[k1], :], ]
        mm = mm + [np.mean(ft[k1], axis=0), ]

    ff = np.concatenate(np.array(ft)-np.array([mm, ]).transpose((1, 0, 2)), axis=0)

    # Normalization techniques
    if 0 == opt:
        ccl = np.cov(ff.T)
    elif 1 == opt:
        ccl = cov_lw(ff)
    elif 2 == opt:
        ccl = cov_oas(ff)
    elif 3 == opt:
        ccl = np.cov(ff.T) + np.eye(ns)
    else:
        raise Exception('opt only accepts  the following values (0,1, 2, 3)')

    try:
        pcl = la.inv(ccl)

        testf = np.stack(ce.x.apply(lambda x: np.diag(x)), 0)
        if normalize:
            testf = testf / testf.sum(axis=1, keepdims=True)
        testf = np.log(testf)

        prp = np.zeros((testf.shape[0], nc))

        for k2 in range(nc):
            prp[:, k2] = np.diag(la.multi_dot((testf - mm[k2], pcl, np.transpose(testf - mm[k2]))))
        labels = cc[np.argmin(prp, axis=1)]
    except np.linalg.LinAlgError:
        # The previous could raise an error if the matrix were singular.
        labels = cc[0]

    return labels


def mdrm(ca, ce):
    """
        Riemaniann Distance Classifier.
        :param ca: Filtered training trials Covariances as DataFrame
        :param ce: Filtered test trials Covariances as DataFrame
        :return: Label of test trials
    """
    cc = np.unique(ca.y)
    try:
        nc = cc.shape[0]
        c1 = df(100*np.ones((ce.shape[0], nc)), columns=range(nc))
        for nk in range(nc):
            center = rm(np.stack(ca.x[ca.y == cc[nk]], 0))
            c1[nk] = ce.x.apply(lambda x: rd(x, center))

        labels = cc[np.argmin(c1.values, axis=1)]
    except np.linalg.LinAlgError:
        labels = cc[0]

    return labels


# Covariance estimators
def cov_lw(data):
    """
    Ledoit and Wolf shrink Covariance estimator
    """
    data = data - np.mean(data, axis=0)
    nd, ns = np.shape(data)
    sn = np.cov(data.T)

    mn = np.trace(sn) / ns
    dn2 = np.sum(np.sum((sn - mn * np.eye(ns)) ** 2))
    bn = 0
    for k1 in range(nd):
        bn = bn + np.sum(np.sum((np.outer(data[k1, :], data[k1, :].T) - sn) ** 2))

    bn2 = np.min(np.array([bn / nd ** 2, dn2]))
    an2 = dn2 - bn2
    y = (mn * bn2 * np.eye(ns) + sn * an2) / dn2
    return y


def cov_trace(x):
    """
    Covariance and trace
    """
    y = np.cov(x)
    return y/np.trace(y)


def cov_oas(data):
    """
    Gaussian shrinkage estimator from "Shrinkage Algorithms for MMSE Covariance Estimation."
    """
    data = data-np.mean(data, axis=0)
    nd, ns = np.shape(data)
    sn = np.cov(data.T)

    t2r = np.trace(np.dot(sn, sn))
    tr2 = np.trace(sn)**2
    ro1 = (tr2-t2r/ns)/((nd-1)/ns*(t2r-tr2/ns))

    ro = np.min(np.array([1, ro1]))
    y = (1-ro)*sn + ro*np.eye(ns)*np.trace(sn)/ns
    return y


def ts_lr(ca, ce):
    """
        Logistic Regression and Tangent Space Classifier.
        :param ca: Filtered training trials Covariances as DataFrame
        :param ce: Filtered test trials Covariances as DataFrame
        :return: Label of test trials
    """
    try:
        center = rm(np.stack(ca.x, axis=0))
        proytr = ts(np.stack(ca.x, axis=0), center)
        proyte = ts(np.stack(ce.x, axis=0), center)
        clf = LogisticRegression(solver='liblinear', multi_class='ovr').fit(proytr, ca.y)
        labels = clf.predict(proyte)
    except np.linalg.LinAlgError:
        labels = np.unique(ca.y)[0]
    return labels
