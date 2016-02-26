import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_image(id):
    object = np.load("mnistTrain.npz")
    
    img = object['images'][id].reshape((28,28)).T
    label = object['labels'][id][0] # 0 is label, 1 is type
    print("labelled as: %s"%label)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) is not 2:
        print("Specify an images id")
        sys.exit(-1)
    identifier = int(sys.argv[1])
    generate_image(identifier)
