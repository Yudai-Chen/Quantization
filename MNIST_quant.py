import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model

from keras.datasets import mnist

# the data, split between train and test sets
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)
print(x_test.shape[0], 'test samples')

total_influence = 0

def cal_influence(weights, num_neighbors):
    global total_influence
    n = len(weights)
    influence = np.copy(weights)
    
    newlist = []
    for i in range(n):
        np.append(newlist, influence[i].flatten())
    sortIdx = np.argsort(newlist)
    newlist = sorted(newlist)

    sizeList = []
    sizeList.append(0)
    for i in range(1, n + 1):
        sizeList.append(sizeList[i - 1] + np.size(influence[i - 1], 0) * np.size(influence[i - 1], 1))
    print(sizeList)
    m = len(newlist)
    for j in range(m):
        neighbor_sum = 0
        neg_num = 0
        pos_num = 0
        neighbor_list = []
        for k in range(1, num_neighbors + 1):
            if j - k >= 0:
                neighbor_list.append(abs(newlist[j - k] - newlist[j]))
            if j + k < m:
                neighbor_list.append(abs(newlist[j + k] - newlist[j]))
        neighbor_list = sorted(neighbor_list)
        neighbor_list = neighbor_list[0:num_neighbors]
        neighbor_sum = sum(neighbor_list)
        neighbor_sum /= num_neighbors
        this_influence = newlist[j] ** 2 / neighbor_sum
        total_influence += this_influence
        
        for i in range(len())
        #layer = 
        #influence[layer][sortIdx[j] // np.size(influence[i][0], 0)][sortIdx[j] % np.size(influence[i][0], 0)] = this_influence
    return influence

def allocate_bits(influence, total_bits):
    global total_influence



def quant(weights, bits):
	n = len(weights)
	new_weights = np.copy(weights)
	for i in range(n):
		if i % 2 == 1:
			tb = 2 * bits
		else:
			tb = bits
		scale = (2 ** (tb - 1) - 1) / max(abs(np.min(weights[i])), abs(np.max(weights[i])))
		new_weights[i] = weights[i] * scale
		new_weights[i] = np.around(new_weights[i])
		# print(new_weights[i])
		new_weights[i] = new_weights[i] / scale
		# print(1 / scale)
		# print(weights[i])

	return new_weights

model = load_model('my_model_nobias.h5')
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

weights = model.get_weights()
print(cal_influence(weights, 10))
print(total_influence)
# model.set_weights(quant(weights, 3))

# score = model.evaluate(x_test, y_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])