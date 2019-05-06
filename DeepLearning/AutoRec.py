import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class AutoRec(object):

    def __init__(self, visibleDimensions, epochs=200, hiddenDimensions=50, learningRate=0.1, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.learningRate = learningRate
        self.batchSize = batchSize
        
                
    def Train(self, X):

        ops.reset_default_graph()

        self.MakeGraph()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        npX = np.array(X)
        
        for epoch in range(self.epochs):

            for i in range(0, npX.shape[0], self.batchSize):
                epochX = npX[i:i+self.batchSize]
                self.sess.run(self.update, feed_dict={self.inputLayer: epochX})

            print("Trained epoch ", epoch)


    def GetRecommendations(self, inputUser):
        
        # Feed through a single user and return predictions from the output layer.
        rec = self.sess.run(self.outputLayer, feed_dict={self.inputLayer: inputUser})

        return rec[0]     

    def MakeGraph(self):

        tf.set_random_seed(0)
        
        # Create varaibles for weights for the encoding (visible->hidden) and decoding (hidden->output) stages, randomly initialized
        self.encoderWeights = {'weights': tf.Variable(tf.random_normal([self.visibleDimensions, self.hiddenDimensions]))}
        self.decoderWeights = {'weights': tf.Variable(tf.random_normal([self.hiddenDimensions, self.visibleDimensions]))}
        
        # Create biases
        self.encoderBiases = {'biases': tf.Variable(tf.random_normal([self.hiddenDimensions]))}
        self.decoderBiases = {'biases': tf.Variable(tf.random_normal([self.visibleDimensions]))}
        
        # Create the input layer
        self.inputLayer = tf.placeholder('float', [None, self.visibleDimensions])
        
        # hidden layer
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.inputLayer, self.encoderWeights['weights']), self.encoderBiases['biases']))
        
        # output layer for our predictions.
        self.outputLayer = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.decoderWeights['weights']), self.decoderBiases['biases']))
       
        # Our "true" labels for training are copied from the input layer.
        self.labels = self.inputLayer
        
        # loss function and optimizer. Try other optimizers, like Adam!
        loss = tf.losses.mean_squared_error(self.labels, self.outputLayer)
        optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(loss)
        
        # What we evaluate each batch.
        self.update = [optimizer, loss]

        
    