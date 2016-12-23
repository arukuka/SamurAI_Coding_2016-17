# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, FunctionSet, cuda
import numpy as np
from gen_action_list import ActionList
import sys

class DQN():

    gamma = 0.99

    def __init__(self):
        self.actions = ActionList()
        
        self.model = FunctionSet(
            conv1_1=L.Convolution2D(4, 8, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(8, 8, 3, stride=1, pad=1),
            conv2_1=L.Convolution2D(8, 16, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(16, 16, 3, stride=1, pad=1),
            fc3=L.Linear(4080, 4096),
            fc4=L.Linear(4096, 4096),
            fc5=L.Linear(4096, self.actions.get_action_size())
        ).to_gpu()
        
    def __call__(self, x):
        y1 = self.model.conv1_1(x)
        y2 = self.model.conv1_2(y1)
        y3 = F.relu(y2)
        y4 = self.model.conv2_1(y3)
        y5 = self.model.conv2_2(y4)
        y6 = F.relu(y5)
        y7 = self.model.fc3(y6)
        y8 = self.model.fc4(y7)
        y9 = F.relu(y8)
        y10 = self.model.fc5(y9)

        return y10

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
        end_flags = np.ndarray(shape=len(rules), dtype=np.bool)
        
        for i in xrange(len(rules)):
            states[i] = np.asarray(rules[i][0], dtype=np.float32)
            actions[i] = rules[i][1]
            rewards[i] = rules[i][2]
            next_states[i] = np.asarray(rules[i][3], dtype=np.float32)
            end_flags[i] = rules[i][4]
        
        Q = self(Variable(cuda.to_gpu(states)))
        next_Q = self(Variable(cuda.to_gpu(next_states)))
        tmp = list(map(np.max, next_Q.data.get()))
        max_next_Q = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(Q.data.get(), dtype=np.float32)
        
        for i in xrange(len(rules)):
            if end_flags[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * max_next_Q[i]
        return F.mean_squared_error(Variable(cuda.to_gpu(target)), Q)

