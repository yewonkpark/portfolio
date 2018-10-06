### Best Model Source Code

### PART 1. Load the dataset
# The dataset is downloadable from https://doi.org/10.1186/s12859-015-0774-y

import sys
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import utils

train_data = pd.read_csv('train_x.csv',header=None)
train_label = pd.read_csv('train_y.csv',header=None)
test_data = pd.read_csv('test_x.csv',header=None)
test_label = pd.read_csv('test_y.csv',header=None)


### PART 2-1. ML-SMOTE (one-time)

from sklearn.neighbors import NearestNeighbors
import scipy
jaccard = scipy.spatial.distance.jaccard

SMOTE1st_train_data = train_data
SMOTE1st_train_label = train_label

D = train_data
L = train_label
L_tp = train_label.T

# calculate the IRLbl indices

IRLbl_base = np.sum(L, axis=0)
IRLbl_max = np.amax(IRLbl_base)
IRLbl_original = [IRLbl_max / each if each > 0 else 1 for each in IRLbl_base]

MaxIR = np.amax(IRLbl_original)
MeanIR = np.mean(IRLbl_original)

minority_count = 0

# define function for generate sample/label set (majority vote for nominal attributes)

def newCommon(valueSum) :
    bar = k/2 # because it already includes the sample itself
    new_Commonset = [1 if each >= bar else 0 for each in valueSum]
    return new_Commonset


# iterate through each label(side-effect) and check if the label is minority
for i in range(0,L.shape[1]) : 
    print("smote for label",i,"th start")
    k=6  # number of neighbours: set to six to pick k = 5 neighbours except for itself. 
    idxBag = np.empty(1, dtype=int)
    minBag = []
    lblBag = []
    
    sum_each = np.sum(L[i], axis=0) # sum of the i-th label (total number of positive labels) 
    IRLbl_each = IRLbl_max/sum_each if sum_each>0 else 1  # calculate the imbalance ratio for i-th label
    
    if IRLbl_each > MeanIR : # if i-th label is monority label
        minority_count += 1
        idx_to_add = np.where(L[i]==1) # to extract the index of rows which contains positive label for the i-th label
        idxBag = np.append(idxBag, idx_to_add)
        idxBag = idxBag[1:]
        for ind in idxBag :
            dat_to_add = D.iloc[ind,:] #store the data for label generation for new synthetic sample
            lbl_to_add = L.iloc[ind,:] #store the label for label generation for new synthetic sample
            minBag.append(dat_to_add)
            lblBag.append(lbl_to_add)
            
            # --------------- now we have minBag and lblBag
            # --------------- search for k-nearest neighbours and perform synthetic sampling             
        if len(minBag) < 6:  #for minority labels with less than 5 neighbours.
            k = len(minBag)
            
        NN = NearestNeighbors(n_neighbors=k, metric= jaccard)

        if k > 1 :
            NN.fit(minBag) # fit the nearest neighbor...
            for sample in minBag : # for each data sample in the minBag, find the index of k neighbours
                KNN = NN.kneighbors([sample])
                KNN_ind = KNN[1] 
                
                # sum of the value (features) of all nearest neighbours and original sample itself
                neigh_feature_sum = np.zeros(D.shape[1])
                for neigh in range (0,len(KNN_ind[0])) :
                    neigh_feature_sum += minBag[KNN_ind[0][neigh]]
                    
                neigh_label_sum = np.zeros(L.shape[1])
                for neigh in range (0,len(KNN_ind[0])) :
                    neigh_label_sum += lblBag[KNN_ind[0][neigh]]
                          
                # generate synthetic sample
                new_sample = newCommon(neigh_feature_sum)
                new_sample_df = pd.DataFrame(new_sample).T
                # generate synthetic label set
                new_label = newCommon(neigh_label_sum)
                new_label_df = pd.DataFrame(new_label).T
                # append the newly generated sample and label to original dataset
                SMOTE1st_train_data = SMOTE1st_train_data.append(new_sample_df, ignore_index=True)
                SMOTE1st_train_label = SMOTE1st_train_label.append(new_label_df, ignore_index=True)

        elif k == 1 : 
    # if there is only one sample in the minBag... no neighbour with same positive minority label thus duplicate
            SMOTE1st_train_data =SMOTE1st_train_data.append(minBag[0], ignore_index=True)
            SMOTE1st_train_label = SMOTE1st_train_label.append(lblBag[0], ignore_index=True)

        else : print('ERROR')
        
print(minority_count)
print(SMOTE1st_train_data.shape)
print(SMOTE1st_train_label.shape)


### PART 2-2. ML-SMOTE  (double-time)


SMOTE2nd_train_data = SMOTE1st_train_data
SMOTE2nd_train_label = SMOTE1st_train_label

D = SMOTE1st_train_data
L = SMOTE1st_train_label
L_tp = SMOTE1st_train_label.T

# calculate the IRLbl indices

IRLbl_base = np.sum(L, axis=0)
IRLbl_max = np.amax(IRLbl_base)
IRLbl_original = [IRLbl_max / each if each > 0 else 1 for each in IRLbl_base]

MaxIR = np.amax(IRLbl_original)
MeanIR = np.mean(IRLbl_original)

minority_count = 0

# iterate through each label(side-effect) and check if the label is minority
for i in range(0,L.shape[1]) : 

    print("smote for label",i,"th start")
    k=6  # number of neighbours: set to six to pick k = 5 neighbours except for itself. 
    idxBag = np.empty(1, dtype=int)
    minBag = []
    lblBag = []
    
    sum_each = np.sum(L[i], axis=0) # sum of the i-th label (total number of positive labels) 
    IRLbl_each = IRLbl_max/sum_each if sum_each>0 else 1  # calculate the imbalance ratio for i-th label
    
    if IRLbl_each > MeanIR : # if i-th label is monority label
        minority_count += 1
        idx_to_add = np.where(L[i]==1) # to extract the index of rows which contains positive label for the i-th label
        idxBag = np.append(idxBag, idx_to_add)
        idxBag = idxBag[1:]
        for ind in idxBag :
            dat_to_add = D.iloc[ind,:] #store the data for label generation for new synthetic sample
            lbl_to_add = L.iloc[ind,:] #store the label for label generation for new synthetic sample
            minBag.append(dat_to_add)
            lblBag.append(lbl_to_add)
            
            # --------------- now we have minBag and lblBag
            # --------------- search for k-nearest neighbours and perform synthetic sampling             
        if len(minBag) < 6:  #for minority labels with less than 5 neighbours.
            k = len(minBag)
            
        NN = NearestNeighbors(n_neighbors=k, metric= jaccard)

        if k > 1 :
            NN.fit(minBag) # fit the nearest neighbor...
            for sample in minBag : # for each data sample in the minBag, find the index of k neighbours
                KNN = NN.kneighbors([sample])
                KNN_ind = KNN[1] 
                
                # sum of the value (features) of all nearest neighbours and original sample itself
                neigh_feature_sum = np.zeros(D.shape[1])
                for neigh in range (0,len(KNN_ind[0])) :
                    neigh_feature_sum += minBag[KNN_ind[0][neigh]]
                    
                neigh_label_sum = np.zeros(L.shape[1])
                for neigh in range (0,len(KNN_ind[0])) :
                    neigh_label_sum += lblBag[KNN_ind[0][neigh]]
                    
                    
                # generate synthetic sample
                new_sample = newCommon(neigh_feature_sum)
                new_sample_df = pd.DataFrame(new_sample).T
                # generate synthetic label set
                new_label = newCommon(neigh_label_sum)
                new_label_df = pd.DataFrame(new_label).T
                # append the newly generated sample and label to original dataset
                SMOTE2nd_train_data = SMOTE2nd_train_data.append(new_sample_df, ignore_index=True)
                SMOTE2nd_train_label = SMOTE2nd_train_label.append(new_label_df, ignore_index=True)

        elif k == 1 : 
    # if there is only one sample in the minBag... no neighbour with same positive minority label thus duplicate
            SMOTE2nd_train_data =SMOTE2nd_train_data.append(minBag[0], ignore_index=True)
            SMOTE2nd_train_label = SMOTE2nd_train_label.append(lblBag[0], ignore_index=True)

        else : print('ERROR')
        

print(minority_count)
print(SMOTE2nd_train_data.shape)
print(SMOTE2nd_train_label.shape)



### PART 3. Define Label Weighted Binary Cross Entropy loss function

### SELECT THE DATASET want to use for training
train_data = SMOTE2nd_train_data # in case of use double-time SMOTE
train_label = SMOTE2nd_train_label # in case of use double-time SMOTE

# create the weight base which is positive sample count for each label

label_num = train_label.shape[1]
attribute_num = train_data.shape[1]
len_train = train_label.shape[0]
len_test = test_label.shape[0]

label_stats = pd.DataFrame(train_label.sum(axis=0), columns=["count"])
label_stats["name"] = label_stats.index.values
label_stats.reset_index(drop=True)

weight_base = label_stats.iloc[:,0]
#print(weight_base.head(10))

mu = 0.35 # this to heuristically adjusted

# generate the weight to be applied to loss function (higher mu; higher penalty for false negative)

weight_array = []
for i in weight_base :
    if i == 0 :
        weight_array.append(1)
    else : 
        a = np.log(mu*len_train/i)
        if a < 1 :
            weight_array.append(1)
        else :
            weight_array.append(a)

            
print(weight_array[0:10]) # check for the first 10 label weights


# define loss function

import keras.backend.tensorflow_backend as tfb
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

pos_weight = weight_array  

def custom_weighted_binary_crossentropy(targets, logits, pos_weight=weight_array, name=None):
    
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), logits.dtype.base_dtype)
    logits = tf.clip_by_value(logits, _epsilon, 1 - _epsilon)
    logits = tf.log(logits / (1 - logits))
    # compute weighted loss

    with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        try:
          targets.get_shape().merge_with(logits.get_shape())
        except ValueError:
          raise ValueError(
          "logits and targets must have the same shape (%s vs %s)" %
          (logits.get_shape(), targets.get_shape()))

        loss = []
        for i in range (0,label_num-1):
            log_weight = 1 + (pos_weight[i] - 1) * targets[i]
            loss_i = math_ops.add((1 - targets[i]) * logits[i],
            log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits[i]))) + nn_ops.relu(-logits[i])), 
                                name=name)
            loss.append(loss_i)
            return tf.reduce_mean(loss)
        
        
### PART 4. Compile the best model with Dropout


from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from __future__ import print_function
from keras.initializers import RandomUniform, Zeros
from keras import regularizers

# make sure the input data is in the right type for Keras
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# dropout parameters
dropout_rate1 = 0.8  # rate to be retained
dropout_rate2 = 0.5
dropout1 = 1-dropout_rate1 # rate to be dropped out
dropout2 = 1-dropout_rate2


# input and output size
i_size = train_data.shape[1]
o_size = train_label.shape[1]

# model architecture: defined by base MLP
hidden_neuron_1 = 500
activation_1 = 'sigmoid'

# adjust batch size if needed
#batch_size = train_data.shape[0]
batch_size = 2048
epochs = 3000

# training stopping condition

class Callback(object):

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class StoppingByLossVal(Callback): # training stops when converged   
    
    def __init__(self, monitor='val_loss', value=0.0001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: stopping THR" % epoch)
            self.model.stop_training = True



model_best = Sequential()
model_best.add(Dropout(dropout1, input_shape=(i_size,)))
model_best.add(Dense(hidden_neuron_1, activation=activation_1, 
                     kernel_initializer='random_normal'
                    ))
model_best.add(Dropout(dropout2))
model_best.add(Dense(o_size, activation='sigmoid')) # sigmoid for multi-label classification



model_best.compile(
    loss = custom_weighted_binary_crossentropy,
    optimizer=Adam(lr=0.001),
    metrics=['binary_accuracy'])
    
model_best.summary()

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

callbacks = [StoppingByLossVal(monitor='val_loss', value=0.001, verbose=1) #stopping condiction
             , es] # EARLY stopping condition
             
history = model_best.fit(train_data, train_label,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(test_data, test_label),
                         shuffle=True,
                         callbacks=callbacks)


### PART5. Output

train_predict_matrix = model_best.predict(train_data, batch_size=batch_size, verbose=1, steps=None)
test_predict_matrix = model_best.predict(test_data, batch_size=batch_size, verbose=1, steps=None)
train_output = pd.DataFrame(train_predict_matrix)
test_output = pd.DataFrame(test_predict_matrix)

# learning curve

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()


### PART6. Thresholding

# constant function
# find the threshold which produces the best F1 score for training data

from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss

bar = np.arange(0,1,0.05)
best_th = 0
best_f1 = 0

for k in range(0,len(bar)) :
    temp_pred = np.zeros(train_predict_matrix.shape)
    temp_pred[train_predict_matrix>=bar[k]] = 1
    temp_pred[train_predict_matrix<bar[k]] = 0
    temp_f1 = f1_score(train_label,temp_pred, average='micro')
    if temp_f1 > best_f1 :
        best_f1 = temp_f1
        best_th = bar[k]

    
print("[best threshold for training result] ",best_th, "with f1 score = ",best_f1)


### PART7. Check the prediction performance

# checking for training set
best_th_1 = best_th 
predict_labels = np.zeros(train_predict_matrix.shape)
bar = best_th_1
predict_labels[train_predict_matrix>=bar] = 1
predict_labels[train_predict_matrix<bar] = 0

# this is the training result
f1score = f1_score(train_label,predict_labels, average='micro')
recall = recall_score(train_label,predict_labels, average='micro')
precision = precision_score(train_label,predict_labels, average='micro')

hloss = hamming_loss(train_label,predict_labels)


print("[training set result with best threshold =",best_th,"f1 :" ,f1score,"recall :",recall, "precision :", precision,"hamming loss :", hloss)


# now apply the best threshold above to the test set
best_th_1 = best_th 
predict_labels = np.zeros(test_predict_matrix.shape)
bar = best_th_1
predict_labels[test_predict_matrix>=bar] = 1
predict_labels[test_predict_matrix<bar] = 0

# this is the test result

f1score = f1_score(test_label,predict_labels, average='micro')
recall = recall_score(test_label,predict_labels, average='micro')
precision = precision_score(test_label,predict_labels, average='micro')

hloss = hamming_loss(test_label,predict_labels)


print("[test set result with best threshold =",best_th,"f1 :" ,f1score,"recall :",recall, "precision :", precision,"hamming loss :", hloss)


# save the prediction result (output probability)
test_output.to_csv('test_outputprob_bestModel.csv', sep=',')

# save the prediction result (thresholded binary label)
predict_labels.to_csv('test_outputprob_bestModel.csv', sep=',')
