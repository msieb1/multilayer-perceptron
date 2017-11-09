import math
import random
import numpy as np
import csv
import re
import matplotlib.pyplot as plt



class NN:
    """
    create a neural networks as class object
    """
    def __init__(self, n_nodes, inputs):
        """

        :param

        n_nodes: array with number of nodes in each layer.
        length(n_nodes - 2) thus equals the number of hidden layers.
        first entry is number of input nodes, last entry output nodes

        inputs: Array of dim=784 vectors
        """

        ############## Hyper parameters ##################
        self.l_rate = 0.1
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.eps = 0.0000001
        self.lambd = 0.1
        self.batch_size = 100


        #######################################
        """
        Here, the following can be specified:
        - activation function (and derivative)
        - output function
        - optimizer (Standard, Momentum, Adam)
        - batch normalization on or off
        """
        self.activation_fun = self.sigmoid
        self.activation_fun_g = self.d_sigmoid
        self.output_fun = self.softmax
        #Specify optimizer: Adam, Standard or Momentum (Standard with Momentum)
        self.optimization_function = 'Standard'
        self.batchnorm_on = False
        self.lrate_decay_on = False

        ######################################

        n = self.batch_size

        self.layers = np.size(n_nodes)-1
        #Parameters W
        self.W = {0: np.zeros(n_nodes[0])}
        self.W_g = {0: np.zeros(n_nodes[0])}
        #bias b
        self.b = {0: np.zeros(n_nodes[0])}
        self.b_g = {0: np.zeros(n_nodes[0])}
        # batch norm parameters gamma and beta
        self.gamma = {0: np.zeros(n_nodes[0])}
        self.gamma_g = {0: np.zeros(n_nodes[0])}
        self.beta = {0: np.zeros(n_nodes[0])}
        self.beta_g = {0: np.zeros(n_nodes[0])}
        #activations a
        self.a = {0: np.zeros((n_nodes[0],n))}
        self.a_g = {0: np.zeros(n_nodes[0])} # gradients just defined for one sample as of now
        #hidden layers h (h[self.layers] := output layer)
        self.h = {0: np.zeros((n_nodes[0],n))}
        self.h_g = {0: np.zeros(n_nodes[0])}
        # batch norm layer including gradient, output, mean and variance of batch
        self.bn = {0: np.zeros((n_nodes[0],n))}
        self.bn_g = {0: np.zeros(n_nodes[0])}
        self.bn_mean = {0: np.zeros(n_nodes[0])}
        self.bn_var = {0: np.zeros(n_nodes[0])}

        # params for early stopping
        self.best_params = {'W': 0, 'b': 0, 'gamma': 0, 'beta': 0, 'best_ii': 0}
        self.best_ii = 0

        for i in np.arange(1, self.layers+1):
            """
            init weights
            """
            self.W[i] = np.zeros((n_nodes[i],n_nodes[i-1]))
            self.W_g[i] = np.zeros((n_nodes[i],n_nodes[i-1]))
            self.b[i] = np.zeros((n_nodes[i],1))
            self.b_g[i] = np.zeros(n_nodes[i])
            self.gamma[i] = np.zeros((n_nodes[i], 1))
            self.beta[i] = np.zeros((n_nodes[i], 1))
            self.gamma_g[i] = np.zeros(n_nodes[i])
            self.beta_g[i] = np.zeros(n_nodes[i])


            self.a[i] = np.zeros((n_nodes[i], n))
            self.bn[i] = np.zeros((n_nodes[i], n))
            self.bn_mean[i] = np.zeros(n_nodes[i])
            self.bn_var[i] = np.zeros(n_nodes[i])
            self.h[i] = np.zeros((n_nodes[i], n))

            self.a_g[i] = np.zeros(n_nodes[i])
            self.bn_g[i] = np.zeros(n_nodes[i])
            self.h_g[i] = np.zeros(n_nodes[i])

            con = np.sqrt(6)/np.sqrt(len(self.h[i])+len(self.h[i-1]))
            row, col = self.W[i].shape
            np.random.seed()
            self.W[i][:, :] = np.reshape(np.random.uniform(-con, con, row*col), [row,col])
            self.b[i][:] = 0.1
            self.gamma[i] = np.zeros((n_nodes[i], 2))
            self.beta[i] = np.zeros((n_nodes[i], 2))
            if i == 1:
             self.W[1] = np.load('W_RBM200.npy')


    def set_hyperparams(self, l_rate, beta_1, beta_2, eps, lambd, batch_size, act, act_g, out, opt, batch_on, lrate_decay_on):
        self.l_rate = l_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.lambd = lambd
        self.batch_size = batch_size
        self.activation_fun = act
        self.activation_fun_g = act_g
        self.output_fun = out
        self.optimization_function = opt
        self.batchnorm_on = batch_on
        self.lrate_decay_on = lrate_decay_on

    def forward_activations(self, inputs):
        """

        :param inputs:
        :return:  yields output as return value
        """
        if ~self.batchnorm_on:
            n_layers = self.layers
            self.h[0] = inputs
            self.a[1] = np.dot(self.W[1], inputs)+ self.b[1]
            self.h[1] = self.activation_fun(self.a[1])

            for i in np.arange(2, n_layers):
                self.a[i] = np.dot(self.W[i], self.h[i-1])+self.b[i]
                self.h[i] = self.activation_fun(self.a[i])

            self.a[n_layers] = np.dot(self.W[n_layers], self.h[n_layers-1])+self.b[n_layers]
            self.h[n_layers] = self.output_fun(self.a[n_layers])
            return self.h[n_layers]


        else:
            n_layers = self.layers
            self.h[0] = inputs
            self.a[1] = np.dot(self.W[1], inputs) + self.b[1]
            self.bn[1], self.bn_mean[1], self.bn_var[1] = self.batchnorm(self.a[1], 1)
            self.h[1] = self.activation_fun(self.bn[1])

            for i in np.arange(2, n_layers):
                self.a[i] = np.dot(self.W[i], self.h[i - 1]) + self.b[i]
                self.bn[i], self.bn_mean[i], self.bn_var[i] = self.batchnorm(self.a[i], i)
                self.h[i] = self.activation_fun(self.bn[i])

            self.a[n_layers] = np.dot(self.W[n_layers], self.h[n_layers - 1]) + self.b[n_layers]
            self.bn[n_layers], self.bn_mean[n_layers], self.bn_var[n_layers] = self.batchnorm(self.a[n_layers], n_layers)
            self.h[n_layers] = self.output_fun(self.bn[n_layers])
            return self.h[n_layers]

    def calculate_loss(self, x_train, y_train):
        try:
            dim, n = x_train.shape
        except:
            dim = len(x_train)
            n = 1
        loss = 0
        weight_loss = 0
        self.forward_activations(x_train)
        output = self.h[self.layers]
        #for i in range(1, self.layers+1):
        #    weight_loss += np.sum(self.W[i]) + np.sum(self.b[i])
        for i in range(0, n):
            ind = np.where(y_train[:,i]==1)
            f = output[ind, i]
            if f < 0.0001:
                f = 0.0001
            loss -= np.log(f)
        return loss/n

    def backprop(self, x_data, y_data):
        """

        :param x_data: single data input
        :param y_data: single data label (one-hot)
        :param index: index of current data point
        :return:
        """

        batch_size = x_data.shape[1]
        if ~self.batchnorm_on:
            n_layers = self.layers
            output = self.h[self.layers]
            e = y_data #serves as indicator function
            W_g = {**self.W_g}
            b_g = {**self.b_g}
            for j in np.arange(batch_size):
                self.a_g[n_layers] = (-(e-output))[:, j]
                for i in np.arange(self.layers, 0, -1):
                    self.W_g[i] = np.outer(self.a_g[i], self.h[i-1][:, j])
                    self.b_g[i] = self.a_g[i]
                    self.h_g[i-1] = np.dot(self.W[i].T, self.a_g[i])
                    self.a_g[i-1] = self.h_g[i-1] * self.activation_fun_g(self.a[i-1][:, j])
                    W_g[i] += self.W_g[i]/batch_size
                    b_g[i] += self.b_g[i]/batch_size
            self.W_g = {**W_g}
            self.b_g = {**b_g}
            return


        else:
            n_layers = self.layers
            output = self.h[self.layers]
            e = y_data #serves as indicator function
            W_g = {**self.W_g}
            b_g = {**self.b_g}
            for j in np.arange(batch_size):
                self.bn_g[n_layers] = (-(e-output))[:, j]
                self.a_g[n_layers], self.gamma_g[n_layers], self.beta_g[n_layers] = self.batchnorm_backward(self.bn_g[n_layers], n_layers)
                for i in np.arange(self.layers, 0, -1):
                    self.W_g[i] = np.outer(self.a_g[i], self.h[i-1][:, j])
                    self.b_g[i] = self.a_g[i]
                    self.h_g[i-1] = np.dot(self.W[i].T, self.a_g[i])
                    self.bn_g[i-1] = self.h_g[i-1] * self.activation_fun_g(self.bn[i-1][:, j])
                    self.a_g[i-1], self.gamma_g[i-1], self.beta_g[i-1] = self.batchnorm_backward(self.bn_g[i-1], i-1)
                    W_g[i] += self.W_g[i]/batch_size
                    b_g[i] += self.b_g[i]/batch_size
            self.W_g = {**W_g}
            self.b_g = {**b_g}
            return

    def update_params(self, m_prev, v_prev, opt_fun):
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        eps = self.eps
        lambd = self.lambd
        l_rate = self.l_rate

        if opt_fun == 'Standard':
            for i in np.arange(1, self.layers+1):
                grad = np.hstack([self.W_g[i], np.atleast_2d(self.b_g[i]).T, self.gamma_g[i], self.beta_g[i]])
                self.W[i] -= l_rate*self.W_g[i]
                self.b[i] -= l_rate*np.atleast_2d(self.b_g[i]).T
                self.gamma[i] -= l_rate*self.gamma_g[i]
                self.beta[i] -= l_rate*self.beta_g[i]
                return m_prev, v_prev

        if opt_fun == 'Momentum':
            for i in np.arange(1, self.layers+1):
                grad = np.hstack([self.W_g[i], np.atleast_2d(self.b_g[i]).T,np.atleast_2d(self.gamma_g[i]).T,np.atleast_2d(self.beta_g[i]).T])
                delta = grad - beta_1*m_prev[i]
                m_prev[i] = delta
                self.W[i] -= l_rate*(delta[:, :-3] + self.lambd*self.W[i])
                self.b[i] -= l_rate*(np.atleast_2d(delta[:, -3]).T + self.lambd*self.b[i])
                self.gamma[i] -= l_rate*(np.atleast_2d(delta[:, -2]).T + self.lambd*self.gamma[i])
                self.beta[i] -= l_rate*(np.atleast_2d(delta[:, -1]).T + self.lambd*self.beta[i])
                return m_prev, v_prev
        # works
        # if opt_fun == 'Adam':
        #     for i in np.arange(1, self.layers+1):
        #         grad = np.hstack([self.W_g[i], np.atleast_2d(self.b_g[i]).T])
        #         rows, cols = grad.shape
        #         m = beta_1*m_prev + (1-beta_1)*grad
        #         #m = delta
        #         v = beta_2*v_prev + (1-beta_2)*grad**2
        #         #m = m / (1-beta_1)
        #         #v = v / (1-beta_2)
        #         self.W[i] -= l_rate*m[:, :-1]/(np.sqrt(v[:, :-1]) + eps)
        #         self.b[i] -= np.atleast_2d(l_rate*(m[:, -1])/(np.sqrt((v[:, -1])) + eps)).T
        #         return m, v
        if opt_fun == 'Adam':
            for i in np.arange(1, self.layers+1):
                grad = np.hstack([self.W_g[i], np.atleast_2d(self.b_g[i]).T,np.atleast_2d(self.gamma_g[i]).T,np.atleast_2d(self.beta_g[i]).T])
                delta = -beta_1*m_prev[i] + (1-beta_1)*grad
                m_prev[i] = delta
                v_prev[i] = beta_2*v_prev[i] + (1-beta_2)*grad**2
                m_prev[i] = m_prev[i] * (1-beta_1) #inquire those
                v_prev[i] = v_prev[i] * (1-beta_2)
                self.W[i] -= l_rate*(m_prev[i][:, :-3]/(np.sqrt(v_prev[i][:, :-3]) + eps)+self.lambd*self.W[i])
                self.b[i] -= l_rate*(np.atleast_2d(m_prev[i][:, -3])/(np.sqrt(v_prev[i][:, -3]) + eps).T + self.lambd*self.b[i])
                self.gamma[i] -= l_rate*(np.atleast_2d(delta[:, -2]).T(np.sqrt(v_prev[i][:, -2]) + eps).T + self.lambd*self.gamma[i])
                self.beta[i] -= l_rate*(np.atleast_2d(delta[:, -1]).T/(np.sqrt(v_prev[i][:, -1]) + eps).T+ self.lambd*self.beta[i])

    def train(self, n_epochs, data):
        """
        train the network with backprop
        :param l_rate:
        :param n_epochs:
        :return:
        """
        opt_fun = self.optimization_function
        l_rate = self.l_rate
        batch_size = self.batch_size
        # m stores gradient, v stores second moment gradient (copying W is just for dictionary dimonesion, aka layers)
        # the actual m and v dict will be filled with all gradients w.r.t W, b, gamma and beta
        m = {**self.W_g}
        v = {**self.W_g}
        for i in range(1, self.layers + 1):
            m[i] = np.hstack([self.W_g[i], np.atleast_2d(self.b_g[i]).T,np.atleast_2d(self.gamma_g[i]).T,np.atleast_2d(self.beta_g[i]).T ])*0
            v[i] = np.hstack([self.W_g[i], np.atleast_2d(self.b_g[i]).T,np.atleast_2d(self.gamma_g[i]).T,np.atleast_2d(self.beta_g[i]).T]) * 0
        x_train = data['x_train']
        y_train = data['y_train']
        x_valid = data['x_valid']
        y_valid = data['y_valid']
        dim, n_samples = x_train.shape


        ####### TRACK LOSS ##############
        loss_tracker = {'train': np.zeros((n_epochs+1, 1)), 'valid': np.zeros((n_epochs+1, 1))}
        loss_tracker['train'][0, 0] = self.calculate_loss(x_train, y_train)
        loss_tracker['valid'][0, 0] = self.calculate_loss(x_valid, y_valid)
        ######################

        ###### TRACK CLASSIFICATION ERROR ##############
        error_tracker = {'train': np.zeros((n_epochs+1, 1)), 'valid': np.zeros((n_epochs+1, 1))}
        self.forward_activations(x_train)
        y_hat = self.h[self.layers]
        error_tracker['train'][0,0] = classification_error(y_hat, y_train)
        self.forward_activations(x_valid)
        y_hat2 = self.h[self.layers]
        error_tracker['valid'][0,0] = classification_error(y_hat2, y_valid)
        ###########################

        ### EARlY STOPPING PARAMS ###
        ii = 0
        jj = 0
        vv = 100000
        pp = 10000
        nn = 10


        for epoch in np.arange(n_epochs):
            randomize = np.arange(x_train.shape[1])
            np.random.shuffle(randomize)
            x_train = x_train[:, randomize][:, :]
            y_train = y_train[:, randomize][:, :]

            for j in np.arange(int(n_samples/batch_size)):
                #get output for current batch
                self.forward_activations((x_train[:, j*batch_size:(j+1)*batch_size])) #Make 784,1 instead of 784,
                self.backprop(x_train[:, j*batch_size:(j+1)*batch_size], y_train[:, j*batch_size:(j+1)*batch_size]) #fix index passing
                m, v = self.update_params(m, v, opt_fun)

                ## EARLY STOPPING
                ii += 1
                if (int(ii / nn) == 1) and (jj != pp):
                    v_prime = self.calculate_loss(x_valid, y_valid)
                    if v_prime < vv:
                        jj = 0
                        self.best_params['W'] = self.W
                        self.best_params['b'] = self.b
                        self.best_params['gamma'] = self.gamma
                        self.best_params['beta'] = self.beta
                        self.best_ii = ii
                        self.best_params['best_ii'] = self.best_ii
                        vv = v_prime
                    else:
                        jj += 1

                ##################################

            if (n_samples -int(n_samples/batch_size)*batch_size) != 0:
                # get data if batch does not divide perfectly sample set
                self.forward_activations((x_train[:, int(n_samples/batch_size)*batch_size:]))  # Make 784,1 instead of 784,
                self.backprop(x_train[:, int(n_samples/batch_size)*batch_size:],
                              y_train[:, int(n_samples/batch_size)*batch_size:])
                m, v = self.update_params(m, v, opt_fun)


                ### EARLY STOPPING ###
                ii += 1
                if (int(ii / nn) == 1) and (jj != pp):
                    ii += nn
                    v_prime = self.calculate_loss(x_valid, y_valid)
                    if v_prime < vv:
                        jj = 0
                        self.best_params['W'] = self.W
                        self.best_params['b'] = self.b
                        self.best_params['gamma'] = self.gamma
                        self.best_params['beta'] = self.beta
                        self.best_ii = ii
                        self.best_params['best_ii'] = self.best_ii
                        vv = v_prime
                    else:
                        jj += 1

                #################

            #update error and loss dictionaries
            loss_tracker['train'][epoch+1, 0] = self.calculate_loss(x_train, y_train)
            loss_tracker['valid'][epoch+1, 0] = self.calculate_loss(x_valid, y_valid)


            y_hat = self.forward_activations(x_valid)
            error_tracker['valid'][epoch+1, 0] = classification_error(y_hat, y_valid)
            y_hat2 = self.forward_activations(x_train)
            error_tracker['train'][epoch + 1, 0] = classification_error(y_hat2, y_train)
            ##################################
            print("Epoch %s: error is %s" % (epoch + 1, loss_tracker['train'][epoch+1, 0]))

            #decrease learning rate if hitting a plateau
            if (np.abs(error_tracker['valid'][epoch+1, 0] - error_tracker['valid'][epoch, 0]) < 0.002) and self.lrate_decay_on:
                self.l_rate /= 1.5


        return loss_tracker, error_tracker

    def batchnorm(self, batch, curr_layer):
        i = curr_layer
        eps = 0.00000001
        dim, n = batch.shape
        m = np.sum(batch, 1)/n
        var = np.var(batch, 1)/n
        x_hat = (batch - m)/(np.sqrt(var + eps))
        y = self.gamma[i]*x_hat + self.beta[i]
        return y, m, var

    def batchnorm_backward(self, dout, curr_layer):
        eps = self.eps
        i = curr_layer
        # get the dimensions of the input/output
        dim, n = dout.shape

        # step9
        dbeta = np.sum(dout, 1)
        dgammax = dout  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * self.bn[i], 1)
        dxhat = dgammax * self.gamma[i]

        # step7
        divar = np.sum(dxhat * self.bn_mean[i], 1)
        dxmu1 = dxhat * 1/self.bn_var[i]

        # step6
        dsqrtvar = -1. / (np.sqrt(self.bn_var[i] + eps) ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(self.bn_var[i] + eps) * dsqrtvar

        # step4
        dsq = 1. / n * np.ones((dim , n)) * dvar

        # step3
        dxmu2 = 2 * self.bn[i] * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, 1)

        # step1
        dx2 = 1. / n * np.ones((dim, n)) * dmu

        # step0
        dx = dx1 + dx2

        return dx, dgamma, dbeta

    #### Activation and Output functions ####

    def sigmoid(self, a):
        np.clip(a,-300,300)
        return 1/(1+np.exp(-a))

    def d_sigmoid(self, a):
        return self.sigmoid(a)*(1-self.sigmoid(a))

    def relu(self, a):
        buff = np.copy(a)
        return np.maximum(a, buff*0)

    def d_relu(self, a):
        res = np.copy(a)
        res[np.where(a > 0)] = 1
        res[np.where(a <= 0)] = 0
        return res

    def tanh(self, a):
        return np.tanh(a)

    def d_tanh(self, a):
        return 1 - np.tanh(a) ** 2

    def softmax(self, a):
        m = np.max(a,0)
        out = np.exp(a - m - np.log(np.sum(np.exp(a - m),0)))
        return out

    #########################################

    def check_gradient(self, eps, layer, ind1, ind2, x_train, y_train):
        W_o = self.W
        b_o = self.b
        real_grad = self.W_g[layer][ind1, ind2]
        self.W[layer][ind1,ind2] += eps
        loss_plus = self.calculate_loss(x_train, y_train)
        self.W[layer][ind1,ind2] -= 2*eps
        loss_minus = self.calculate_loss(x_train, y_train)
        grad = (loss_plus - loss_minus)/(2*eps)
        error = np.abs(grad-real_grad)
        if real_grad > 0.1:
            a = 0
        return error, grad


##########################

def load_data():
    with open('/home/max/PyCharm/PycharmProjects/10-707/hw1/data/digitstrain.txt') as f:
        reader = csv.reader(f, delimiter=" ")
        d = list(reader)
        x_train = np.zeros((784,3000))
        y_train = np.zeros((10,3000))
        for i in range(0, len(d)):
            s = d[i][0]
            p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
            data = [float(i) for i in p.findall(s)]  # Convert strings to float
            p = re.compile(r'\d+')  # Compile a pattern to capture float values
            label = [int(i) for i in p.findall(s)]
            x_train[:,i] = data
            y_train[label[-1],i] = 1
            np.save('x_train.npy', x_train)
            np.save('y_train.npy', y_train)

    with open('/home/max/PyCharm/PycharmProjects/10-707/hw1/data/digitstest.txt') as f:
        reader = csv.reader(f, delimiter=" ")
        d = list(reader)
        x_test = np.zeros((784, 3000))
        y_test = np.zeros((10, 3000))
        for i in range(0, len(d)):
            s = d[i][0]
            p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
            data = [float(i) for i in p.findall(s)]  # Convert strings to float
            p = re.compile(r'\d+')  # Compile a pattern to capture float values
            label = [int(i) for i in p.findall(s)]
            x_test[:, i] = data
            y_test[label[-1], i] = 1
            np.save('x_test.npy', x_test)
            np.save('y_test.npy', y_test)

    with open('/home/max/PyCharm/PycharmProjects/10-707/hw1/data/digitsvalid.txt') as f:
        reader = csv.reader(f, delimiter=" ")
        d = list(reader)
        x_valid = np.zeros((784, 1000))
        y_valid = np.zeros((10, 1000))
        for i in range(0, len(d)):
            s = d[i][0]
            p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
            data = [float(i) for i in p.findall(s)]  # Convert strings to float
            p = re.compile(r'\d+')  # Compile a pattern to capture float values
            label = [int(i) for i in p.findall(s)]
            x_valid[:, i] = data
            y_valid[label[-1], i] = 1
            np.save('x_valid.npy', x_valid)
            np.save('y_valid.npy', y_valid)
    dat = {'x_train': x_train}
    dat['y_train'] = y_train
    dat['x_test'] = x_test
    dat['y_test'] = y_test
    dat['x_valid'] = x_valid
    dat['y_valid'] = y_valid

    return dat

def plotting(path, save_to_file=False):
    path_to_save = path
    plt.figure(1)
    plt.clf()
    plt.plot(np.arange(1,n_epochs+1),loss_tracker['train'][1:], 'ro-', markersize=4, label='train')
    plt.plot(np.arange(1,n_epochs+1),loss_tracker['valid'][1:], 'yo-', markersize=4, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.title('cross-entropy')
    plt.legend(loc='upper left')
    if save_to_file:
        name = 'graph_loss_RBM200'
        path = path_to_save + name
        plt.savefig(path)

    plt.figure(2)
    plt.clf()
    plt.plot(np.arange(1,n_epochs+1),error_tracker['train'][1:], 'bo-', markersize=4, label='train')
    plt.plot(np.arange(1,n_epochs+1),error_tracker['valid'][1:], 'go-', markersize=4, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('error')
    #plt.title('classification error')
    plt.legend(loc='upper left')
    if save_to_file:
        name = 'graph_error_RBM200'
        path = path_to_save + name
        plt.savefig(path)
    #plt.show()

def visualize_params(net):
    #make sure you got good params!
    plt.figure(1)
    plt.clf()
    for i in range(0, 100):
        plt.subplot(10, 10, i+1)
        buff = np.reshape(net.W[1][i, :], [28, 28])
        plt.imshow(buff, cmap='gray')
        plt.tick_params(axis='both', left='off', right='off', which='both', bottom='off', top='off', labelbottom='off', labeltop='off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    name = 'visualized_params2'
    path_to_save = '/home/max/PyCharm/PycharmProjects/10-707/hw1/figures/'
    path = path_to_save + name
    plt.savefig(path)

    an = 0

def normalize(arr):
    delta = 0.00000001
    arr_norm = (arr - np.array([np.mean(arr, 1)]).T) / (np.array([np.std(arr, 1)]).T + delta)  # normalize data
    return arr_norm

def classification_error(prediction, label):
    try:
        n = label.shape[1]
    except:
        n = 1
    row, col = prediction.shape
    prediction += np.reshape(np.random.uniform(-0.0001, 0.0001, row * col), [row, col])
    def array_row_intersection(a, b):
        tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
        return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]

    ind_pred = np.argwhere(prediction == np.max(prediction, axis=0))
    ind_corr = np.argwhere(label)
    n_correct = array_row_intersection(ind_pred, ind_corr).shape[0]
    return (n-n_correct)/n


####################################
if __name__ == '__main__':

	# ONLY LOAD IF .npy files not stored!!
	# data = load_data()

	#normalize if needed
	x_train = (np.load('x_train.npy'))
	y_train = (np.load('y_train.npy'))
	x_valid = (np.load('x_valid.npy'))
	y_valid = (np.load('y_valid.npy'))
	x_test = (np.load('x_test.npy'))
	y_test = (np.load('y_test.npy'))
	# store data in a dictionary
	data = {'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid, 'x_test': x_test, 'y_valid': y_valid}

	######################################

	for i in range(8,9):

	    #specifiy net architecture: .e.g.: 784 inputs, first hidden layer 500 nodes, second one 300 nodes, output 10
	    net = NN([784,100,10], x_train)
	    # specify initial learning rate
	    l_rate = 0.1
	    # momentum parameter
	    beta_1 = 0.9
	    # RMSProp/Adam parameter
	    beta_2 = 0.98
	    # variance square root regularizer
	    eps = 0.0000001
	    # regularizer
	    lambd = 0.1
	    # batch sample size
	    batch_size = 64
	    # activation function: net.sigmoid, net.relu, net.tanh
	    activation_fun = net.sigmoid
	    # activation function derivative: net.d_sigmoid, net.d_relu, net.d_tanh
	    activation_fun_g = net.d_sigmoid
	    # output function: net.softmax
	    output_fun = net.softmax
	    # Specify optimizer: Adam, Standard or Momentum (Standard with Momentum)
	    optimization_function = 'Momentum'
	    # batch normalization on or off
	    batchnorm_on = False
	    # learning rate decay on or off (divides main learning rate by 1.5 if validation error stays constant
	    lrate_decay_on = True
	    # number of training epochs
	    n_epochs = 100
	    # path to save figure and data
	    path = '/home/max/PyCharm/PycharmProjects/10-707/hw2/figures/pretraining/'

	    ####################################

	    # initialize network and execute training
	    net.set_hyperparams(l_rate, beta_1, beta_2, eps, lambd, batch_size, activation_fun, activation_fun_g, output_fun,
	                        optimization_function, batchnorm_on, lrate_decay_on)
	    loss_tracker, error_tracker = net.train(n_epochs, data)

	    # calculate loss on test set
	    # loss_train = net.calculate_loss(x_train, y_train)
	    # out = net.h[net.layers]
	    # err_train = classification_error(out, y_train)
	    # loss_valid = net.calculate_loss(x_valid, y_valid)
	    # out = net.h[net.layers]
	    # err_valid = classification_error(out, y_valid)
	    # loss_test = net.calculate_loss(x_test, y_test)
	    # out = net.h[net.layers]
	    # err_test = classification_error(out, y_test)
	    #
	    # np.save(path + 'err_train.npy', err_train)
	    # np.save(path + 'loss_train.npy', loss_train)
	    # np.save(path + 'err_valid.npy', err_valid)
	    # np.save(path + 'loss_valid.npy', loss_valid)
	    # np.save(path + 'err_test.npy', err_test)
	    # np.save(path + 'loss_test.npy', loss_test)
	    # np.save(path + 'W.npy', net.W[1])

	    plotting(path)
	    # visualize_params(net)
	    plt.show()
