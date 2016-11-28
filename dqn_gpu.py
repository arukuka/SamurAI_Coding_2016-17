# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, cuda
import numpy as np
from gen_action_list import ActionList
import sys

class DQN(chainer.Chain):

    alpha = 0.01
    gamma = 0.99

    def __init__(self):
        super(DQN, self).__init__()
        
        self.actions = ActionList()
        
        links = [('conv1_1', L.Convolution2D(4, 64, 3, stride=1, pad=1))]
        links += [('conv1_2', L.Convolution2D(64, 64, 3, stride=1, pad=1))]
        links += [('_mpool1', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('conv2_1', L.Convolution2D(64, 128, 3, stride=1, pad=1))]
        links += [('conv2_2', L.Convolution2D(128, 128, 3, stride=1, pad=1))]
        links += [('_mpool2', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('conv3_1', L.Convolution2D(128, 256, 3, stride=1, pad=1))]
        links += [('conv3_2', L.Convolution2D(256, 256, 3, stride=1, pad=1))]
        links += [('_mpool3', F.MaxPooling2D(2, 2, 0, True, True))]
        links += [('fc4', L.Linear(1536, 4096))]
        links += [('_dropout4', F.Dropout(0.5))]
        links += [('fc5', L.Linear(4096, 4096))]
        links += [('_dropout5', F.Dropout(0.5))]
        links += [('fc6', L.Linear(4096, self.actions.get_action_size()))]

        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)

        self.forward = links

    def __call__(self, x):
        for name, f in self.forward:
            x = f(x)

        return x

    def next_action(self, state_, epsilon):
        state = np.asanyarray(np.array(state_).reshape(1, 4, 17, 15), dtype=np.float32)
        s = Variable(cuda.to_gpu(state))
        Q = self(s).data
        
        if np.random.rand() < epsilon:
            index = np.random.randint(0, self.actions.get_action_size())
        else:
            index = np.argmax(Q)
        
        return index, self.actions.get_action_str(index)

    def get_loss(self, rules):
        states = np.ndarray(shape=(len(rules), 4, 17, 15), dtype=np.float32)
        actions = np.ndarray(shape=len(rules), dtype=np.uint32)
        rewards = np.ndarray(shape=len(rules), dtype=np.float32)
        next_states = np.ndarray(shape=(len(rules), 4, 17, 15), dtype=np.float32)
        
        for i in xrange(len(rules)):
            states[i] = np.asarray(rules[i][0], dtype=np.float32)
            actions[i] = rules[i][1]
            rewards[i] = rules[i][2]
            next_states[i] = np.asarray(rules[i][3], dtype=np.float32)
        
        Q = self(Variable(cuda.to_gpu(states))).data
        next_Q = self(Variable(cuda.to_gpu(next_states))).data
        tmp = list(map(np.max, next_Q))
        max_next_Q = np.asanyarray(tmp, dtype=np.float32)
        target = Q.copy()
        
        for i in xrange(len(rules)):
            target[i][actions[i]] = rewards[i] + self.gamma * max_next_Q[i]
        return F.mean_squared_error(Variable(target), Variable(Q))

