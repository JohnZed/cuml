# Trivial test to see differences in two umap runs
import numpy as np
from sklearn import datasets
from cuml.manifold import UMAP

data, labels = datasets.make_blobs(100, 10, centers=5)

args = dict(verbose = 0, random_seed = 42)
umap1 = UMAP(**args)
umap2 = UMAP(**args)

print("----- First pass ------")
embedding1 = umap1.fit_transform(data)
print("----- Second pass ------")
embedding2 = umap2.fit_transform(data)
print("----- Done        ------")

print("First embeddings:")
print(embedding1[:2,:])
print("Second embeddings:")
print(embedding2[:2,:])

print("Mean abs diff: ",
      np.abs(embedding2 - embedding1).mean())
