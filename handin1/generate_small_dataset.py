from shared import load_full_data, save_data
import numpy as np

def get_subset_of_data(data):
    # select 100 entries with label '7' and 100 entries with label '2'
    N = 1000 # number of samples to select
    
    idx_seven = (data['labels']==[7]).nonzero()[0][0:N]
    idx_two = (data['labels']==[2]).nonzero()[0][0:N]
    
    images_seven = data['images'][idx_seven]
    images_two = data['images'][idx_two]
    images = np.concatenate((images_seven, images_two))
    
    labels_seven = np.zeros(N)+7
    labels_two = np.zeros(N)+2
    labels = np.concatenate((labels_seven, labels_two))
    
    return {'images': images,
            'labels': labels}

def run():
    data = load_full_data()
    small_data = get_subset_of_data(data)
    save_data("small_data", **small_data) #images=small_data['images'], labels=small_data['labels'])

if __name__ == "__main__":
    run()

