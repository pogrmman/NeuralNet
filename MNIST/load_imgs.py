import gzip
import pickle

with gzip.open("mnist_data.pkl.gz") as f:
    data = pickle.Unpickler(f).load()

for seq in data:
    for i, item in enumerate(seq):
        for j, num in enumerate(item[0]):
            item[0][j] = num / 255

train, test, val = data
