# This Code has been developed by Javier Olias.
import scipy.io as sio
import sys
from pandas import DataFrame as df
import numpy as np


import Auxiliary_functions as jf
import Normalization_functions as norm

seed = 5
numberOfTraining = (40,)  # Number of training trials. It admits a list and give results for each case
numberOfTest = (40,)  # Number of test trials
numberOfMcSimulations = 10  # Monte Carlo simulations
numberOfFilters = 6

# The following lines are to store the codes that generated each result
with open(sys.argv[0], 'r') as aux:
    mainCode = aux.read()
with open(str(jf).split('\'')[-2], 'r')as aux:
    auxFunctionsCode = aux.read()

# DataFrame to store the results
results = df({
    'Accuracy': [],
    'Classifier': [],
    'numberOfTraining': [],
    'trTrials': [],
    'Artifacts': [],
    'Normalization': []
})
# This initializations speeds up the codes
partialResults2 = results.copy()
partialResults1 = results.copy()


# We are considering a unique file.
dataMatrix = sio.loadmat('Data.mat')
for k0 in numberOfTraining:
    for k30 in range(0, numberOfMcSimulations):
        # Split and mix the data for the test and training.
        trainingTrials, testTrials, trueTestClasses, ta, te = jf.split_data(dataMatrix, numberOfTest[0], k0, seed + k30)

        trainingClass = np.unique(trainingTrials.y)
        trainingLength, testLength = trainingTrials.y.shape[0], len(trueTestClasses)
        sampleLength, sensorLength = trainingTrials.x[0].shape
        trainingCovs = df({'x': trainingTrials.x.apply(lambda x1: np.cov(x1.T)), 'y': trainingTrials.y})
        testCovariances = df({'x': testTrials.x.apply(lambda x1: np.cov(x1.T))})

        normalizedMeanCov, normalizationCov = norm.normalization_cov(np.stack(trainingTrials.x), 10, 0.001)

        trainingNormCovs = df({
                'x': trainingTrials.x.apply(lambda x1: norm.normalize_covariance(x1, normalizationCov)[0]),
                'y': trainingTrials.y
            })
        testNormCovs = df({
                'x': testTrials.x.apply(lambda x1: norm.normalize_covariance(x1, normalizationCov)[0])
            })

        trainingFilteredCovs, testFilteredCovs = \
            jf.csp_function(trainingNormCovs, numberOfFilters, testNormCovs, tr=False)

        stdTrainingFilteredCovs, stdTestFilteredCovs = \
            jf.csp_function(trainingCovs, numberOfFilters, testCovariances, tr=True)

        computedClass = jf.lda_clasfier(trainingFilteredCovs, testFilteredCovs, 0, normalize=True)

        # Storing the results
        partialResults1 = partialResults1.append(df({
            'Accuracy': np.where(np.array(trueTestClasses) == np.array(computedClass), 1, 0).tolist(),
            'Classifier': 'LDA',
            'numberOfTraining': [ta, ] * numberOfTest[0],
            'trTrials': te.tolist(),
            'Normalization': 'Yes'
        }), ignore_index=True, sort=False)

        computedClass = jf.lda_clasfier(stdTrainingFilteredCovs, stdTestFilteredCovs, 0, normalize=True)

        # Storing the results
        partialResults1 = partialResults1.append(df({
            'Accuracy': np.where(np.array(trueTestClasses) == np.array(computedClass), 1, 0).tolist(),
            'Classifier': 'LDA',
            'numberOfTraining': [ta, ] * numberOfTest[0],
            'trTrials': te.tolist(),
            'Normalization': 'No'
        }), ignore_index=True, sort=False)

        computedClass = jf.ts_lr(trainingFilteredCovs, testFilteredCovs)

        # Storing the results
        partialResults1 = partialResults1.append(df({
            'Accuracy': np.where(np.array(trueTestClasses) == np.array(computedClass), 1, 0).tolist(),
            'Classifier': 'TSLR',
            'numberOfTraining': [ta, ] * numberOfTest[0],
            'trTrials': te.tolist(),
            'Normalization': 'Yes'
        }), ignore_index=True, sort=False)

        computedClass = jf.ts_lr(stdTrainingFilteredCovs, stdTestFilteredCovs)

        # Storing the results
        partialResults1 = partialResults1.append(df({
            'Accuracy': np.where(np.array(trueTestClasses) == np.array(computedClass), 1, 0).tolist(),
            'Classifier': 'TSLR',
            'numberOfTraining': [ta, ] * numberOfTest[0],
            'trTrials': te.tolist(),
            'Normalization': 'No'
        }), ignore_index=True, sort=False)

    # Storing results in upper level
    partialResults2 = partialResults2.append(partialResults1, ignore_index=True, sort=False)
    partialResults1 = results.copy()


# Results Representation
numberOfTrials = dataMatrix['DATA']['y'][0][0].shape[0]
trueClass = np.reshape(dataMatrix['DATA']['y'][0][0], numberOfTrials)
classes = np.unique(trueClass)
data = [item[0] for item in
        np.split(np.transpose(dataMatrix['DATA']['x'][0][0], (2, 0, 1)), numberOfTrials, axis=0)]

TrueCovClass = np.transpose(dataMatrix['DATA']['TrueCovClass'][0][0], (2, 0, 1))

normalizedMeanCov, normalizationCov = norm.normalization_cov(np.stack(data), 10, 0.001)
normalizedCovs = np.stack(list(map(lambda x1: norm.normalize_covariance(x1, normalizationCov)[0], data)))
trialCovs = np.stack(list(map(lambda x1: np.cov(x1.T), data)))

a = np.zeros(len(classes))
b = np.zeros(len(classes))
for k in range(len(classes)):
    normalizedClassesCovs = np.mean(normalizedCovs[np.where(trueClass == classes[k])[0], :, :], axis=0)
    a[k] = norm.scale_invar_Riemannian_distance(normalizedClassesCovs, TrueCovClass[k, :, :])
    ClassesCovs = np.mean(trialCovs[np.where(trueClass == classes[k])[0], :, :], axis=0)
    b[k] = norm.scale_invar_Riemannian_distance(ClassesCovs, TrueCovClass[k, :, :])


print('\nMean scale-inv. Riemannian dist.(EstimCovClass||TrueCovClass) before normalization: %.2f' % b.mean())

print('Mean scale-inv. Riemannian dist.(EstimCovClass||TrueCovClass)  after normalization: %.2f' % a.mean())

to_show = partialResults2.groupby(['Classifier', 'Normalization']).Accuracy.mean().unstack()  # [nSel.file[nk]]
# To print partial results
print(
    '''
                      Classical     Normalized
    Classifier
    LDA              %.4f         %.4f
    TSLR             %.4f         %.4f'''
    % (to_show.No['LDA'], to_show.Yes['LDA'], to_show.No['TSLR'], to_show.Yes['TSLR']))
