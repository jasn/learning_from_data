import numpy as np
from main import classify_k_wise

data = np.load('mnistTest.npz')
images = np.concatenate((data['images'],np.ones((data['images'].shape[0],1))), axis=1)
labels = data['labels'].reshape(-1,1)


def c2(theta):
    classifications = classify_k_wise(theta, images)
    errors = (classifications != labels)
    return errors.sum(),len(labels)

d = np.load('trained_multinomial.npz')
trained = d['trained']
indices = [(dic['lambda'], c2(dic['theta'])) for dic in trained]

for x in trained:
    errs,n = c2(x['theta'])
    print('%.2f'%(errs/n*100),'%.0e'%x['lambda'],errs,n)
