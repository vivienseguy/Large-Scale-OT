import numpy as np
from keras import optimizers
from keras import backend as K
from keras.models import Input, Model, Sequential
from keras.layers import Dense
import tensorflow as tf
import os
import time


class StochasticSemiDiscreteTransport:

    def __init__(self, reg_type='entropy', reg_val=0.1, model_potential_fun=None, xt=None, wt=None):

        self.fixed_point_iteration = 0

        self.model_potential_fun = model_potential_fun

        self.reg_type = reg_type
        self.reg_val = reg_val

        self.Xt = xt
        self.wt = wt
        self.nt = xt.shape[0]
        self.d_input = xt.shape[1]
        self.features_size = self.d_input

        # In case we want to learn OT between the data and a Gaussian which is fitted to the data
        self.xt_mean = np.mean(xt, axis=0)
        self.gaussian_mean = self.xt_mean
        (U,S,V) = np.linalg.svd(np.cov(np.transpose(xt)))
        self.L = np.sqrt(S)[:, None]*V


    def build_dual_OT_model(self):

        Xt = K.constant(self.Xt)
        v0 = np.zeros((self.nt,))

        self.Xsi = K.placeholder(dtype='float64', shape=(None, self.d_input,))
        self.i_t = K.placeholder(dtype='int32',shape=(None,))
        self.Xti = tf.gather(Xt, self.i_t, name='xt')

        self.u = self.model_potential_fun(self.d_input, name='u_layer_%d' % (self.fixed_point_iteration)) # dual variable u parameterized as a NN
        self.v = K.variable(v0) # discrete second dual variable

        self.ui = self.u(self.Xsi)
        self.vi = K.gather(self.v, self.i_t)

        self.D = K.reshape(K.sum(K.square(self.Xsi),axis=1), (-1, 1))+K.reshape(K.sum(K.square(self.Xti),axis=1), (1, -1))-2.*K.dot(self.Xsi, K.transpose(self.Xti)) # squared L2 norm ground cost

        self.lossi = K.sum(self.ui)*K.cast(K.shape(self.vi)[0], dtype='float64') + K.sum(self.vi)*K.cast(K.shape(self.ui)[0], dtype='float64') - self.reg_val*K.sum(K.exp((K.reshape(self.ui, (-1, 1)) + K.reshape(self.vi, (1, -1)) - self.D)/self.reg_val))

        self.gui = K.gradients(self.lossi, self.u.trainable_weights)
        self.gvi = K.gradients(self.lossi, [self.vi])

        self.Xsi_input = Input(tensor=self.Xsi)
        self.model_u = Model([self.Xsi_input], [self.u(self.Xsi_input)])


    def fit_from_gaussian(self, lr=0.01, epochs=10, batch_size=100, mean=None, std=None, processor_type='cpu', processor_index="1"):

        self.build_dual_OT_model()

        nb_batch_per_epoch = max([int((self.nt*self.nt)/float(batch_size * batch_size)), 1])

        # We compute the gradient update explicitly since the Keras has trouble with the gather node needed to update the discrete variable v
        updates = [tf.add(self.u.trainable_weights[i], lr*self.gui[i]/(batch_size*batch_size)) for i in range(len(self.u.trainable_weights))]
        updates.append(tf.scatter_add(self.v, self.i_t, lr*self.gvi[0]/(batch_size*batch_size)))
        update = K.function([self.Xsi, self.i_t], [self.lossi], updates=updates)

        self.losses = []
        self.time = []
        history = {}

        tic = time.time()

        self.init = tf.global_variables_initializer()
        config = getConfig(processor_index, log_device_placement=False)
        self.sess = tf.Session(config=config)

        with self.sess as sess, tf.device('/' + processor_type + ':0'):

            sess.run(self.init)

            for e in range(epochs):

                lossb = np.zeros((nb_batch_per_epoch,))

                for b in range(nb_batch_per_epoch):

                    if mean==None:
                        xs_batch = self.gaussian_mean + np.dot(
                            np.random.normal(size=(batch_size * self.d_input)).reshape((batch_size, self.d_input)), self.L)
                    else:
                        xs_batch = mean + std*np.random.normal(size=(batch_size * self.d_input)).reshape((batch_size, self.d_input))

                    i2 = np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)
                    lossb[b] = update((xs_batch, i2))[0]

                    # print('batch loss = %10.3f', lossb[b])

                print('Epoch : {}, loss = {}'.format(e+1, np.sum(lossb)))

                self.losses.append(-np.sum(lossb) / (self.nt*self.nt))
                self.time.append(time.time()-tic)

            self.vfin = sess.run(self.v)

            history['losses'] = self.losses
            history['time'] = self.time

            return history

    def build_barycentric_mapping_model(self):

        Xt = K.constant(self.Xt)
        self.i_t = K.placeholder(dtype='int32', shape=(None,), name='target_inds')
        self.Xsi = K.placeholder(dtype='float64', shape=(None, self.d_input,))
        self.Xti = tf.gather(Xt, self.i_t, name='xt')

        self.v = K.constant(self.vfin)

        self.ui = self.u(self.Xsi)
        self.vi = K.gather(self.v, self.i_t)

        self.D = K.reshape(K.sum(K.square(self.Xsi), axis=1), (-1, 1)) + K.reshape(
            K.sum(K.square(self.Xti), axis=1), (1, -1)) - 2. * K.dot(self.Xsi, K.transpose(self.Xti))

        self.barycentric_mapping_layer = get_sample_mapping(self.d_input)
        self.barycentric_mapping = self.barycentric_mapping_layer(self.Xsi)
        self.barycentric_mapping_test = self.barycentric_mapping_layer(K.placeholder(dtype='float64', shape=(None, self.d_input), name='xs_test'))

        H = K.exp((K.reshape(self.ui, (-1, 1)) + K.reshape(self.vi, (1, -1)) - self.D) / self.reg_val)

        self.Db = K.reshape(K.sum(K.square(self.barycentric_mapping), axis=1), (-1, 1)) + K.reshape(
            K.sum(K.square(self.Xti), axis=1), (1, -1)) - 2. * K.dot(self.barycentric_mapping, K.transpose(self.Xti))
        self.loss_barycentric = K.sum(H*self.Db)


    def fit_barycentric_mapping_from_gaussian(self, lr=0.01, epochs=10, batch_size=100, mean=None, std=None, processor_type='cpu', processor_index="1"):

        self.build_barycentric_mapping_model()

        nb_batch_per_epoch = max([int(self.nt*self.nt/float(batch_size * batch_size)), 1])

        config = getConfig(processor_index, log_device_placement=False)
        self.barycentric_learning_sess = tf.Session(config=config)

        self.init_op = tf.global_variables_initializer()

        train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss_barycentric, var_list=self.barycentric_mapping_layer.trainable_weights)

        self.losses = []
        self.time = []
        history = {}

        tic = time.time()

        with self.barycentric_learning_sess, tf.device('/' + processor_type + ':0'):

            self.barycentric_learning_sess.run(self.init_op)

            for e in range(epochs):

                lossb = np.zeros((nb_batch_per_epoch,))

                for b in range(nb_batch_per_epoch):

                    if mean==None:
                        xs_batch = self.gaussian_mean + np.dot(
                            np.random.normal(size=(batch_size * self.d_input)).reshape((batch_size, self.d_input)), self.L)
                    else:
                        xs_batch = mean + std*np.random.normal(size=(batch_size * self.d_input)).reshape((batch_size, self.d_input))

                    i2 = np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt).astype(
                        np.int32)

                    self.barycentric_learning_sess.run(train_step, feed_dict={self.Xsi: xs_batch, self.i_t: i2})
                    lossb[b] = self.barycentric_learning_sess.run(self.loss_barycentric,
                                                                  feed_dict={self.Xsi: xs_batch, self.i_t: i2})

                print('Epoch : {}, loss={}'.format(e + 1, np.sum(lossb)))

                self.losses.append(np.sum(lossb)/(self.nt*self.nt))
                self.time.append(time.time()-tic)

            saver = tf.train.Saver()
            save_path = saver.save(self.barycentric_learning_sess, "tmp/model.ckpt")

            print("Model saved in path: %s" % save_path)

        history['losses'] = self.losses
        history['time'] = self.time

        return history


    def predict_v(self):
        return self.vfin


    def predict_f(self, x_s):

        with tf.Session().as_default() as sess:
            new_saver = tf.train.import_meta_graph('tmp/model.ckpt.meta')
            new_saver.restore(sess, 'tmp/model.ckpt')

            Xs = sess.graph.get_tensor_by_name("xs_test:0")
            f = sess.graph.get_tensor_by_name("barycentric_mapping_1/output_layer/MatMul:0")

            f_val = sess.run(f, feed_dict={Xs: x_s})

        return f_val





def getConfig(visible_device_list, log_device_placement = False):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = visible_device_list
    config.log_device_placement = log_device_placement
    return config


def get_sample_mapping(d):
    u = Sequential(name='barycentric_mapping')
    u.add(Dense(512, input_dim=d, activation='relu'))
    u.add(Dense(512, input_dim=d, activation='relu'))
    u.add(Dense(d, activation='linear', use_bias=False, name='output_layer'))
    return u