from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

X, y = make_classification(n_classes=3, class_sep=3,
                           weights=[0.1, 0.7, 0.2], n_informative=3, n_redundant=2, flip_y=0,
                           n_features=10, n_clusters_per_class=1, n_samples=10, random_state=10)
print X
print y
print('\nOriginal dataset shape {}'.format(Counter(y)))

rus = RandomUnderSampler(ratio='all', random_state=1)

X_1, y_1 = rus.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_1)))

X_res, y_res = rus.fit_sample([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]], [0, 0, 0, 1, 2, 3])
print('Resampled dataset shape {}'.format(Counter(y_res)))
print y_res