## RBM pre-training experiments (1-layer Deep Belief Network)


#### PART0: Load the dataset
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


#### PART1: Define the RBM model
# the code is based on the source code can be found in https://gist.github.com/blackecho/db85fab069bd2d6fb3e7
# modifications were mainly made to the storing parameters for each training epochs

from scipy import misc

def sample_prob(probs, rand):
    """ Takes a tensor of probabilities (as from a sigmoidal activation)
    and samples from all the distributions
    :param probs: tensor of probabilities
    :param rand: tensor (of the same shape as probs) of random values
    :return : binary sample of probabilities
    """
    return tf.nn.relu(tf.sign(probs - rand))


def gen_batches(data, batch_size):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]



class RBM(object):


    def __init__(self, num_visible, num_hidden, visible_unit_type='bin', 
                 main_dir='/home/enterprise.internal.city.ac.uk/acvn710', model_name='rbm_model',
                 gibbs_sampling_steps=1, learning_rate=0.01, momentum = 0.9, l2 = 0.001, batch_size=10, 
                 num_epochs=10, stddev=0.1, verbose=0, plot_training_loss=True):

        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param visible_unit_type: type of the visible units (binary or gaussian)
        :param main_dir: main directory to put the models, data and summary directories
        :param model_name: name of the model, used to save data
        :param gibbs_sampling_steps: optional, default 1
        :param learning_rate: optional, default 0.01
        :param momentum: momentum for gradient descent, default 0.9
        :param l2: l2 weight decay, default 0.001
        :param batch_size: optional, default 10
        :param num_epochs: optional, default 10
        :param stddev: optional, default 0.1. Ignored if visible_unit_type is not 'gauss'
        :param verbose: level of verbosity. optional, default 0
        :param plot_training_loss: whether or not to plot training loss, default True
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.main_dir = main_dir
        self.model_name = model_name
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2 = l2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev
        self.verbose = verbose

        self._create_model_directory()
        self.model_path = os.path.join(self.main_dir, self.model_name)
        self.plot_training_loss = plot_training_loss

        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.dw = None
        self.dbh_ = None 
        self.dbv_ = None
        self.hiddenprob_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None
        self.recontruct = None

        self.loss_function = None
        self.batch_cost = None
        self.batch_free_energy = None

        self.training_losses = []
        self.hiddenprob = []
        self.weights = []
        self.hbias = []
        self.vbias = []
        self.d_weight = []
        self.d_hbias = []
        self.d_vbias = []
        self.input_data = None
        self.hrand = None
        self.validation_size = None

        self.tf_session = None
        self.tf_saver = None

    def fit(self, train_set, validation_set=None, restore_previous_model=False):

        """ Fit the model to the training data.
        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        if validation_set is not None:
            self.validation_size = validation_set.shape[0]

        tf.reset_default_graph()

        self._build_model()

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            
            self.tf_saver.save(self.tf_session, self.model_path)

            if self.plot_training_loss:
                plt.plot(self.training_losses)
                plt.title("Training batch losses v.s. iteractions")
                plt.xlabel("Num of training iteractions")
                plt.ylabel("Reconstruction error")
                plt.show()

  
        
    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        """

        init_op = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

    def _train_model(self, train_set, validation_set):

        """ Train the model.
        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :return: self
        """
        np.random.shuffle(train_set) # moved to here from _run_train_step (in original code) 
        #to shuffle the train data only at epoch 0 and keep track of the change on each units


        for i in range(self.num_epochs):
            self._run_train_step(train_set)
            if i % 10 == 0 :
                self.weights.append(self.W.eval())
                self.hbias.append(self.bh_.eval())
                self.vbias.append(self.bv_.eval())               
                self.d_weight.append(self.dw.eval(feed_dict=self._create_feed_dict(train_set)))
                self.d_hbias.append(self.dbh_.eval(feed_dict=self._create_feed_dict(train_set)))
                self.d_vbias.append(self.dbv_.eval(feed_dict=self._create_feed_dict(train_set)))
                self.hiddenprob.append(self.hiddenprob_.eval(feed_dict=self._create_feed_dict(train_set)))

            if validation_set is not None:
                self._run_validation_error(i, validation_set)


                                

    def _run_train_step(self, train_set):

        """ Run a training step. A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch. If self.plot_training_loss 
        is true, will record training loss after each batch. 
        :param train_set: training set
        :return: self
        """

        batches = [_ for _ in gen_batches(train_set, self.batch_size)]
        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]
        

        for batch in batches:

            if self.plot_training_loss:
                _, loss = self.tf_session.run([updates, self.loss_function], 
                                              feed_dict=self._create_feed_dict(batch))
                self.training_losses.append(loss)

            else:
                self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch))
                
    
        
    def _run_validation_error(self, epoch, validation_set):

        """ Run the error computation on the validation set and print it out for each epoch. 
        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """

        loss = self.tf_session.run(self.loss_function,
                                   feed_dict=self._create_feed_dict(validation_set))

        if self.verbose == 1:
            print("Reconstruction error at step %s: %s" % (epoch, loss))
            


        
    def _create_feed_dict(self, data):

        """ Create the dictionary of data to feed to TensorFlow's session during training.
        :param data: training/validation set batch
        :return: dictionary(self.input_data: data, self.hrand: random_uniform)
        """

        return {
            self.input_data: data,
            self.hrand: np.random.rand(data.shape[0], self.num_hidden),
        }

    def _build_model(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.
        :return: self
        """

        self.input_data, self.hrand = self._create_placeholders()
        self.W, self.bh_, self.bv_, self.dw, self.dbh_, self.dbv_, self.hiddenprob_ = self._create_variables()

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        nn_input = vprobs

        for step in range(self.gibbs_sampling_steps - 1):
            hprobs, hstates, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(nn_input)
            nn_input = vprobs

        self.recontruct = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs1  # encoded data, used by the transform method
        encode = hprobs1
        
        self.hiddenprob_ = tf.reshape(encode,[771,self.num_hidden])

        #tf.assign_add(ref,value) : Update 'ref' by adding 'value' to it.
        dw = positive - negative
        self.dw = (self.learning_rate*(self.momentum*self.dw + (1-self.momentum)*dw)) - self.learning_rate*self.l2*self.W
        self.w_upd8 = self.W.assign_add(self.dw)
        #self.dw = self.momentum*self.dw + (1-self.momentum)*dw
        #self.w_upd8 = self.W.assign_add(self.learning_rate*self.dw - self.learning_rate*self.l2*self.W)

        dbh_ = tf.reduce_mean(hprobs0 - hprobs1, 0)
        self.dbh_ = self.momentum*self.dbh_ + self.learning_rate*dbh_
        self.bh_upd8 = self.bh_.assign_add(self.dbh_)

        dbv_ = tf.reduce_mean(self.input_data - vprobs, 0)
        self.dbv_ = self.momentum*self.dbv_ + self.learning_rate*dbv_
        self.bv_upd8 = self.bv_.assign_add(self.dbv_)


# reconstruction error
        self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs)))        
        self.batch_cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs), 1))   
        self._create_free_energy_for_batch()
        


        
    def _create_free_energy_for_batch(self):

        """ Create free energy ops to batch input data 
        :return: self
        """

        if self.visible_unit_type == 'bin':
            self._create_free_energy_for_bin()    
        elif self.visible_unit_type == 'gauss':
            self._create_free_energy_for_gauss()
        else:
            self.batch_free_energy = None

    def _create_free_energy_for_bin(self):

        """ Create free energy for mdoel with Bin visible layer
        :return: self
        """


        self.batch_free_energy = - (tf.matmul(self.input_data, tf.reshape(self.bv_, [-1, 1])) + \
                                    tf.reshape(tf.reduce_sum(tf.log(tf.exp(tf.matmul(self.input_data, self.W) + self.bh_) + 1), 1), [-1, 1]))

    def _create_free_energy_for_gauss(self):

        """ Create free energy for model with Gauss visible layer 
        :return: self
        """

        self.batch_free_energy = - (tf.matmul(self.input_data, tf.reshape(self.bv_, [-1, 1])) - \
                                    tf.reshape(tf.reduce_sum(0.5 * self.input_data * self.input_data, 1), [-1, 1]) + \
                                    tf.reshape(tf.reduce_sum(tf.log(tf.exp(tf.matmul(self.input_data, self.W) + self.bh_) + 1), 1), [-1, 1]))

# a placeholder operation that must be fed with data on execution.

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: tuple(input(shape(None, num_visible)), 
                       hrand(shape(None, num_hidden)))
        """

        x = tf.placeholder('float', [None, self.num_visible], name='x-input')
        hrand = tf.placeholder('float', [None, self.num_hidden], name='hrand')
        


        return x, hrand

# start with random weights by normal distribution of average 0, zeros for all bias

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: tuple(weights(shape(num_visible, num_hidden),
                       hidden bias(shape(num_hidden)),
                       visible bias(shape(num_visible)))
        """

        W = tf.Variable(tf.random_normal((self.num_visible, self.num_hidden), mean=0.0, stddev=0.01), name='weights')
        dw = tf.Variable(tf.zeros([self.num_visible, self.num_hidden]), name = 'derivative-weights')

        bh_ = tf.Variable(tf.zeros([self.num_hidden]), name='hidden-bias')
        dbh_ = tf.Variable(tf.zeros([self.num_hidden]), name='derivative-hidden-bias')

        bv_ = tf.Variable(tf.zeros([self.num_visible]), name='visible-bias')
        dbv_ = tf.Variable(tf.zeros([self.num_visible]), name='derivative-visible-bias')
        
        hiddenprob_ = tf.Variable(tf.zeros([771,self.num_hidden]), name='hidden-probability', dtype=tf.float32)


        return W, bh_, bv_, dw, dbh_, dbv_ , hiddenprob_

# with sigmoid activation function

    def gibbs_sampling_step(self, visible):

        """ Performs one step of gibbs sampling.
        :param visible: activations of the visible units
        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """

        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.
        :param visible: activations of the visible units
        :return: tuple(hidden probabilities, hidden binary states)
        """
        
        hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)

        hstates = sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: visible probabilities
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)
        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible, hidden_probs, hidden_states):

        """ Compute positive associations between visible and hidden units.
        :param visible: visible units
        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states
        :return: positive association = dot(visible.T, hidden)
        """

        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def _create_model_directory(self):

        """ Create the directory for storing the model
        :return: self
        """

        if not os.path.isdir(self.main_dir):
            print("Created dir: ", self.main_dir)
            os.mkdir(self.main_dir)

    def getRecontructError(self, data):

        """ return Reconstruction Error (loss) from data in batch.
        :param data: input data of shape num_samples x visible_size
        :return: Reconstruction cost for each sample in the batch
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_loss = self.tf_session.run(self.batch_cost,
                                             feed_dict=self._create_feed_dict(data))
            return batch_loss

    def getFreeEnergy(self, data):

        """ return Free Energy from data.
        :param data: input data of shape num_samples x visible_size
        :return: Free Energy for each sample: p(x)
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_FE = self.tf_session.run(self.batch_free_energy,
                                           feed_dict=self._create_feed_dict(data))

            return batch_FE
        
        
    def getRecontruction(self, data): # get final reconstruction (v') for each data

        with tf.Session() as self.tf_session:
            
            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_reconstruct = self.tf_session.run(self.recontruct, 
                                                    feed_dict=self._create_feed_dict(data))

            return batch_reconstruct


    def getHiddenlayer(self, data): # get final encode(hidden layer units probability) for each data

        with tf.Session() as self.tf_session:
            
            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_hiddenlayer = self.tf_session.run(self.encode, 
                                                    feed_dict=self._create_feed_dict(data))
            

            return batch_hiddenlayer 

### Load a trained model from disk
    def load_model(self, shape, gibbs_sampling_steps, model_path):

        """ Load a trained model from disk. The shape of the model
        (num_visible, num_hidden) and the number of gibbs sampling steps
        must be known in order to restore the model.
        :param shape: tuple(num_visible, num_hidden)
        :param gibbs_sampling_steps:
        :param model_path:
        :return: self
        """

        self.num_visible, self.num_hidden = shape[0], shape[1]
        self.gibbs_sampling_steps = gibbs_sampling_steps
        
        tf.reset_default_graph() # clears the default graph stack and resets the global default graph

        self._build_model()

        init_op = tf.global_variables_initializer() # An Op that initializes global variables in the graph
        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)
            self.tf_saver.restore(self.tf_session, model_path)

            
### this bit is for getting weights/bias matrix afterwards!

    def get_model_parameters(self, data):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """
        
        
        with tf.Session() as self.tf_session: # release resources when they are no longer required

            self.tf_saver.restore(self.tf_session, self.model_path) # save/restore


            return {
                'W': self.W.eval(),
                'bh_': self.bh_.eval(),
                'bv_': self.bv_.eval(),
                'weights' : tf.stack(self.weights).eval(),
                'hbias':tf.stack(self.hbias).eval(),
                'vbias' : tf.stack(self.vbias).eval(),
                'd_weights' : tf.stack(self.d_weight).eval(),
                'd_hbias' : tf.stack(self.d_hbias).eval(),
                'd_vbias' : tf.stack(self.d_vbias).eval(),
                'hiddenprob' : tf.stack(self.hiddenprob).eval()
            }
        

#### PART2: Train the RBM

num_visible_train = train_data.shape[1]


RBM_ext = RBM(num_visible = num_visible_train, num_hidden = 500, learning_rate = 0.00001, batch_size = 10,
           num_epochs=5000, verbose=1, momentum = 0.3, l2 = 0, model_name='rbm_extended')


RBM_ext.fit(train_data, validation_set=train_data)

RBM_ext_parameter_set = RBM_ext.get_model_parameters(train_data)
RBM_ext_result_w = RBM_ext_parameter_set['W']
RBM_ext_result_h_bias = RBM_ext_parameter_set['bh_']
RBM_ext_result_v_bias = RBM_ext_parameter_set['bv_']


#print(RBM_ext_parameter_set['weights'].shape)
#print(RBM_ext_parameter_set['vbias'].shape)
#print(RBM_ext_parameter_set['hbias'].shape)
#print(RBM_ext_parameter_set['hiddenprob'].shape)
#print(RBM_ext_parameter_set['d_weights'].shape)
#print(RBM_ext_parameter_set['d_vbias'].shape)
#print(RBM_ext_parameter_set['d_hbias'].shape)


#### PART 4: visualise the RBM training process
# based on the paper : Yosinski, J. and Lipson, H., 2012, July. Visually debugging restricted boltzmann machine training with a 
# 3d example. In Representation Learning Workshop, 29th International Conference on Machine Learning



plot_epoch = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 499]
select_batch_start = 680
select_batch_end =690


# 1) probability of hidden activation : plot the probability [0,1] for each hidden neuron 
  """ for each example within A MINI BATCH as greyscale value of an image
    each row = for a given input sample, each hidden neuron's activation
    each column = for a given hidden neuron, activiation for each input sample
    top : before traininig, : initialisation checking (grey) 
    middle : after one mini-batch .... : if not converging smoothly cos of the high lr, will FLICKER, 
    end : after converged
    should be no distinct horizontal/vertical line if learning is working properly.
    the plot can show if some hidden units are never used or if some smaple activates an unsually large/small number of hidden units """


# create loop for showing the changes of hidden unit probability for a selected batch

for i in plot_epoch :   
    batch = RBM_ext_parameter_set['hiddenprob'][i]   
    select_viz = batch[select_batch_start:select_batch_end]
    if i == 0 : print("training epoch = ", i+1)
    else : print("training epoch = ", i*10)
    plt.figure(figsize=(200, 10))
    plt.imshow(select_viz, cmap='gray' ,aspect='auto')
    plt.show()


# 2) weight histogram : histogram of the weight updates and a histogram of the weights for a selected mini-batch
# top three plots to display a histogram of values in vbias, W, hbias
# bottom three plots to display a histogram of the most recent updates to the vbias, W, hbias
# mm = mean absolute value of magnitude : according to the paper, mm of bottom three should be  1/1000 the weights
# for bias, updates can be bigger

def plotit(values) :
    plt.hist(values)
    plt.title('mm = %g' % np.mean(np.fabs(values)))

for i in plot_epoch : 
    #plt.figure(figsize=(100,50))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(plt.title, fontsize=30)
    if i == 0 : print("training epoch = ", i+1)
    else : print("training epoch = ", i*10)

    plt.subplot(231); plotit(RBM_ext_parameter_set['vbias'][i][select_batch_start:select_batch_end])
    plt.subplot(232); plotit(RBM_ext_parameter_set['weights'][i][select_batch_start:select_batch_end].flatten())
    plt.subplot(233); plotit(RBM_ext_parameter_set['hbias'][i][select_batch_start:select_batch_end])
    plt.subplot(234); plotit(RBM_ext_parameter_set['d_vbias'][i][select_batch_start:select_batch_end])
    plt.subplot(235); plotit(RBM_ext_parameter_set['d_weights'][i][select_batch_start:select_batch_end].flatten())
    plt.subplot(236); plotit(RBM_ext_parameter_set['d_hbias'][i][select_batch_start:select_batch_end])
    plt.show()

    
#### PART 5: extract the weights and bias from pre-trained RBM and transform them into the format which can be fed into the MLP 

RBM_ext_result_w = np.asarray(RBM_ext_result_w, dtype=None)
RBM_ext_result_h_bias = np.asarray(RBM_ext_result_h_bias, dtype=None)

weight_matrix_RBM_ext_ = []
weight_matrix_RBM_ext_.append(RBM_ext_result_w)
weight_matrix_RBM_ext_.append(RBM_ext_result_h_bias)




#### PART 6: fine-tuning phase (base MLP)

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



model_withRBM = Sequential()
model_withRBM.add(Dropout(dropout1, input_shape=(i_size,)))
model_withRBM.add(Dense(hidden_neuron_1, activation=activation_1, 
                     kernel_initializer='random_normal'
                    ))
model_withRBM.add(Dropout(dropout2))
model_withRBM.add(Dense(o_size, activation='sigmoid')) # sigmoid for multi-label classification

#feed the pre-trained weights and bias from RBMs
model_WITH.layers[0].set_weights(weight_matrix_RBM_ext_)


model_withRBM.compile(
    loss = custom_weighted_binary_crossentropy,
    optimizer=Adam(lr=0.001),
    metrics=['binary_accuracy'])
    
model_withRBM.summary()

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

callbacks = [StoppingByLossVal(monitor='val_loss', value=0.001, verbose=1) #stopping condiction
             , es] # EARLY stopping condition
             
history = model_withRBM.fit(train_data, train_label,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(test_data, test_label),
                         shuffle=True,
                         callbacks=callbacks)


### PART7. Output

train_predict_matrix = model_withRBM.predict(train_data, batch_size=batch_size, verbose=1, steps=None)
test_predict_matrix = model_withRBM.predict(test_data, batch_size=batch_size, verbose=1, steps=None)
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


### PART8. Thresholding

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


### PART9. Check the prediction performance

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
test_output.to_csv('test_outputprob_DBNModel.csv', sep=',')

# save the prediction result (thresholded binary label)
predict_labels.to_csv('test_outputprob_DBNModel.csv', sep=',')
   
