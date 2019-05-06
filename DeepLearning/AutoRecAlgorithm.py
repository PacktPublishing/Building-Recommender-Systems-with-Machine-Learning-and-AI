# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:08:25 2018

@author: Frank
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from AutoRec import AutoRec

class AutoRecAlgorithm(AlgoBase):

    def __init__(self, epochs=100, hiddenDim=100, learningRate=0.01, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        numUsers = trainset.n_users
        numItems = trainset.n_items
        
        trainingMatrix = np.zeros([numUsers, numItems], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            trainingMatrix[int(uid), int(iid)] = rating / 5.0
        
        # Create an RBM with (num items * rating values) visible nodes
        autoRec = AutoRec(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        autoRec.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = autoRec.GetRecommendations([trainingMatrix[uiid]])
            
            for itemID, rec in enumerate(recs):
                self.predictedRatings[uiid, itemID] = rec * 5.0
        
        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
    