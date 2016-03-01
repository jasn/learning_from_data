import numpy as np
from main import classify


def c(params):
    f = (labels == params['first_digit']) | (labels == params['second_digit'])
    i = images[f.ravel()]
    l = params['first_digit'] == labels[f.ravel()]
    return classify(params['theta'], np.concatenate((i, np.ones((i.shape[0], 1))), axis=1)), l, params['lambda']


def c2(params):
    r, lab, lam = c(params)
    errors = (r != lab).sum()
    return errors, len(lab)


data = np.load('mnistTest.npz')
d = np.load('trained_pairs.npz')
labels = data['labels']
images = data['images']
trained = d['trained']
indices = [np.argmin([c2(v)[0] for v in row]) for row in trained]
print(indices,len(indices),len(trained))
for i, row in zip(indices, trained):
    a,b = c2(row[i])
    print('%.2f%%'%(a/b*100), row[i]['first_digit'], row[i]['second_digit'],'%.0e'%row[i]['lambda'], (a,b))
