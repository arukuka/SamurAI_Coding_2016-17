# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, FunctionSet, cuda
import numpy as np
from gen_action_list import ActionList
import sys

class DQN():

    def __init__(self):
        self.models = []
        for i in xrange(3):
          model = FunctionSet(
              conv1=L.Convolution2D(4, 8, 3, stride=1, pad=1),
              conv2=L.Convolution2D(8, 8, 3, stride=1, pad=1),
              fc3=L.Linear(2040, 512),
              fc4=L.Linear(512, 512),
              fc5=L.Linear(512, 15 * 15)
          )
          self.models.append(model)
        
    def __call__(self, x):
        ys = []
        for i in xrange(3):
            y = self.models[i].conv1(x)
            y = self.models[i].conv2(y)
            y = F.relu(y)
            y = self.models[i].fc3(y)
            y = self.models[i].fc4(y)
            y = F.relu(y)
            y = self.models[i].fc5(y)
            ys.append(y)

        return ys[0], ys[1], ys[2]

    def next_action(self, state_):
        state = np.asanyarray(np.array(state_).reshape(1, 4, 17, 15), dtype=np.float32)
        s = Variable(cuda.to_gpu(state))
        Q0, Q1, Q2 = self(s).data
        P0 = F.softmax(Q0)
        P1 = F.softmax(Q1)
        P2 = F.softmax(Q2)
        
        index0 = np.argmax(P0)
        index1 = np.argmax(P1)
        index2 = np.argmax(P2)
        
        return index0, P0[index], index1, P1[index2], index2, P2[index2]

    def get_loss(self, rules):
        states = np.ndarray(shape=(len(rules), 4, 17, 15), dtype=np.float32)
        positions0 = np.ndarray(shape=len(rules), dtype=np.uint32)
        positions1 = np.ndarray(shape=len(rules), dtype=np.uint32)
        positions2 = np.ndarray(shape=len(rules), dtype=np.uint32)
        
        for i in xrange(len(rules)):
            states[i] = np.asarray(rules[i][0], dtype=np.float32)
            positions0[i] = rules[i][1]
            positions1[i] = rules[i][2]
            positions2[i] = rules[i][3]
        
        Q0, Q1, Q2 = self(Variable(cuda.to_gpu(states)))
        arr = np.asanyarray(F.softmax(Q0).data.get()[0].copy(), dtype=np.float32)
        arr.sort()
        pos = np.argmax(cuda.to_cpu(Q0.data.get()[0]))
        print >> sys.stderr, "{}, {} ({}) ... <=> {}, {}".format(pos % 15, pos / 15, arr[::-1][0:5], positions0[0] % 15, positions0[0] / 15)
        target0 = np.zeros(shape=(len(rules)), dtype=np.int32)
        target1 = np.zeros(shape=(len(rules)), dtype=np.int32)
        target2 = np.zeros(shape=(len(rules)), dtype=np.int32)
        
        for i in xrange(len(rules)):
            target0[i] = positions0[i]
            target1[i] = positions1[i]
            target2[i] = positions2[i]
            
        
        print >> sys.stderr, "\tacc0 : {}".format(float(cuda.to_cpu(F.accuracy(Q0, Variable(cuda.to_gpu(target0))).data)))
        print >> sys.stderr, "\tacc1 : {}".format(float(cuda.to_cpu(F.accuracy(Q1, Variable(cuda.to_gpu(target1))).data)))
        print >> sys.stderr, "\tacc2 : {}".format(float(cuda.to_cpu(F.accuracy(Q2, Variable(cuda.to_gpu(target2))).data)))
        return F.softmax_cross_entropy(Q0, Variable(cuda.to_gpu(target0))), F.softmax_cross_entropy(Q1, Variable(cuda.to_gpu(target1))), F.softmax_cross_entropy(Q2, Variable(cuda.to_gpu(target2)))

