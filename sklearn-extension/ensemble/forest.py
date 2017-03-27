import numpy as np

from sklearn.ensemble.forest import _generate_unsampled_indices, RandomForestClassifier
from sklearn.datasets import make_classification

from itertools import combinations
import matplotlib.pyplot as plt

def proximities(forest, X):	
	n_samples, _ = X.shape

	prox_matrix = np.zeros((n_samples, n_samples))
	norm = np.zeros((n_samples))
	
	# TODO: parallelize it
	for i, estimator in enumerate(forest.estimators_):
		unsampled_indices = np.arange(n_samples)
		leafs = estimator.apply(X[unsampled_indices, :])
		# complexity...
		# worst case : O(n^2 + nlogn)
		# avg case: O(nlogn + ?)		
		# best case : O(nlogn + n)		
		order = np.argsort(leafs)
		i = 0
		while i < leafs.shape[0]:
			leaf = leafs[order[i]] 
			j = i
			while j < leafs.shape[0] and leaf == leafs[order[j]]:
				k = j + 1
				while k < leafs.shape[0] and leaf == leafs[order[k]]:
					prox_matrix[unsampled_indices[order[j]],
					 unsampled_indices[order[k]]] += 1
					prox_matrix[unsampled_indices[order[k]],
					 unsampled_indices[order[j]]] += 1
					
					k += 1
				j += 1

			i = j

	np.fill_diagonal(prox_matrix, forest.n_estimators)
	return prox_matrix / forest.n_estimators


def imputing_missing_values(forest, X, y, max_iter = 5):
	n_samples, n_features = X.shape

	classes = np.unique(y)
	
	print(X)
	missing = np.isnan(X)
	Xt = X.copy()
	for i in range(n_features):
		col = Xt[:, i]
		for c in classes:
			col[np.logical_and(missing[:, i], (y == c))] = \
			 col[np.logical_and(~missing[:, i], (y == c))].mean()
		Xt[:, i] = col

	for i in range(max_iter):
		forest.fit(Xt, y)
		prox = proximities(forest, Xt)
		for i in range(n_features):
			non_missing_prox = prox[missing[:, i], :][:, ~missing[:, i]].T
			
			Xt[missing[:, i], i] = \
			 np.dot(Xt[~missing[:, i], i], non_missing_prox) / non_missing_prox.sum(0)

	return Xt

if __name__ == '__main__':
	
	# missing = np.random.choice(np.arage((10, 2)), size=3)
	X, y = make_classification(n_samples = 100, n_features = 2, n_redundant = 0, random_state = 0)

	X_test, y_test = make_classification(n_samples = 10, n_features = 2, n_redundant = 0, random_state = 10)
	rf = RandomForestClassifier(max_features='auto', n_estimators = 100, random_state = 0, n_jobs=-1)
	rf.fit(X,y)

	matrix = proximities(rf, X_test)

	print(matrix)
	fig, ax = plt.subplots()
	c = np.array(['blue', 'red'])
	print(y[1])
	ax.scatter(X[1, 0], X[1, 1], c = c[y[1]])
	# ax.scatter(X[:, 0], X[:, 1], c = y)
	# ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)

	# for i in range(X_test.shape[0]):
	# 	ax.annotate(i, (X_test[i, 0], X_test[i, 1]))
	
	# plt.show()
	print(X)
	arr = np.array([[1,0]])
	X[arr[:,0],arr[:,1]] = np.nan
	Xt = imputing_missing_values(rf, X, y, max_iter = 6)

	ax.scatter(Xt[:, 0], Xt[:, 1], c = c[y])
	# ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)

	for i in range(Xt.shape[0]):
		ax.annotate(i, (Xt[i, 0], Xt[i, 1]))
	
	plt.show()

