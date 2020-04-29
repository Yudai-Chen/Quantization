import keras
import numpy as np
import copy
from keras.models import load_model
from keras.utils import print_summary, to_categorical

from keras.datasets import cifar10

# the data, split between train and test sets
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_test = to_categorical(y_test, num_classes)
x_test = x_test.astype('float32')
x_test /= 255.0

total_influence = 0
num_weight = 0
max_weight = 0
min_weight = 0

def cal_influence(weights, num_neighbors):
    global total_influence
    n = len(weights)
    weights = copy.deepcopy(weights)
    newlist = []
    for i in range(n):
        if len(np.shape(weights[i])) == 1:  # skip layers with 1-dimesnion weights
            continue
        newlist = np.concatenate((newlist, weights[i].flatten()))
    newlist = sorted(newlist)

    influence_list = []
    m = len(newlist)
    for j in range(m):  # for each weight
        neighbor_sum = 0
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

        influence_list.append(this_influence)

    # rehshape
    influence = []
    count_size = 0
    for i in range(n):
        layer_shape = np.shape(weights[i])
        if len(layer_shape) == 1:
            continue
        else:
            layer_size = 1
            for size in layer_shape:
                layer_size *= size

            layer_list = influence_list[count_size : count_size + layer_size]
            count_size += layer_size
            layer_weights = np.reshape(np.array(layer_list), layer_shape)
            influence.append(layer_weights)

    return influence

def allocate_bits(influence, total_bits):
    global total_influence
    for i in range(len(influence)):
        layer_bits_allo = []
        for influ in influence[i].flatten():
            layer_bits_allo.append(total_bits * influ // total_influence + 2)
        influence[i] = np.reshape(np.array(layer_bits_allo), np.shape(influence[i]))
    return influence


# def uniform_allocate_bits(influence, bits_per_weight):
#     global total_influence
#     for i in range(len(influence)):
#         for x in range(len(influence[i])):
#             for y in range(len(influence[i][0])):
#                 influence[i][x][y] = bits_per_weight
#     return influence

def quant(weights, allocation):
    global max_abs
    n = len(weights)
    new_weights = []
    for i in range(n):
        if len(np.shape(weights[i])) == 1: 
            new_weights.append(weights[i])
        else:
            layer_allo = allocation[round(i/2)].flatten()
            layer_weights = weights[i].flatten()
            for j in range(layer_weights.size):
                scale = (2 ** (layer_allo[j] - 1) - 1) / max_abs
                layer_weights[j] = layer_weights[j] * scale
                layer_weights[j] = np.around(layer_weights[j])
                layer_weights[j] = layer_weights[j] / scale
            layer_weights = np.reshape(layer_weights, np.shape(weights[i]))
            new_weights.append(layer_weights)

    return new_weights

AVE_BITS_PER_WEIGHT = 8

model = load_model('models/keras_cifar10_model.h5')
score = model.evaluate(x_test, y_test)
print('Original test loss:', score[0])
print('Original test accuracy:', score[1])

weights = model.get_weights()
for i in range(len(weights)):
    num_weight += weights[i].size
    max_weight = np.max(weights[i])
    min_weight = np.min(weights[i])
    max_abs = max(abs(max_weight), abs(min_weight))
influence = cal_influence(weights, 10)
allocation = allocate_bits(influence, (AVE_BITS_PER_WEIGHT-2) * num_weight)
quantilized = quant(weights, allocation)
model.set_weights(quantilized)

score = model.evaluate(x_test, y_test)
print('Our quantization test loss:', score[0])
print('Our Quantization test accuracy:', score[1])

# # Uniform Quantization
# allocation = uniform_allocate_bits(influence, AVE_BITS_PER_WEIGHT)
# quantilized = quant(weights, allocation)
# model.set_weights(quantilized)

# score = model.evaluate(x_test, y_test)
# print('Uniform Quantization test loss:', score[0])
# print('Uniform Quantization test accuracy:', score[1])