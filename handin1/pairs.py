from shared import load_full_data, save_data
import numpy as np
import main as m
import datetime

def main():
    lambdas = [10**i for i in range(-6,2)] + [0]
    full_data = load_full_data()
    images_all = full_data['images']
    labels_all = full_data['labels']

    thetas = {}

    trained = []
    
    for i in range(10):
        for j in range(i+1,10):
            now = datetime.datetime.now().strftime('%c')
            #print(f"[{datetime.datetime.now().strftime('%c')}] digits {i} vs. {j}")
            print("[%s] digits %d vs %d"%(now, i, j))
            filter = ((labels_all == i) | (labels_all == j))[:, 0]
            images = images_all[filter]
            labels = ((labels_all[filter]) == i)
            n,d = images.shape
            X = np.concatenate((images, np.ones((images.shape[0],1))),axis=1)
            y = labels
            phi = np.random.uniform(size=(d+1, 1))
            thetas = [{'first_digit':i,
                       'second_digit':j,
                       'theta': m.learn_stochastic(X, y, l),
                       'lambda': l}
                      for l in lambdas]
            trained.append(thetas)
            save_data("trained.npz", trained=trained)

    
    save_data("trained.npz", trained=trained)
    
if __name__ == "__main__":
    main()
