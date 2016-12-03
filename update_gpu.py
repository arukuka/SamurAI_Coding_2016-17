# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import optimizers, serializers, cuda
from dqn_gpu import DQN
import os.path
import sys
import os
import pickle

turn_path = os.path.join(os.path.dirname(__file__), 'turn/')

files = os.listdir(turn_path)
rules = []
for fn in files:
    with open(turn_path + fn, "rb") as f:
        r = pickle.load(f)
        rules.append(r)
    # os.remove(turn_path + fn)   

dat_path = os.path.join(os.path.dirname(__file__), 'dat/')

cuda.get_device(0).use()
model = DQN().to_gpu(0)

optim = optimizers.Adam()
optim.setup(model)

if os.path.isfile(dat_path + "dqn.model"):
    print >> sys.stderr, "::loading dqn.model..."
    serializers.load_hdf5(dat_path + "dqn.model", model)

if os.path.isfile(dat_path + "dqn.state"):
    print >> sys.stderr, "::loading dqn.state...."
    serializers.load_hd5f(dat_path + "dqn.state", optim)

for epoch in xrange(100):
    print "Epoch: {}".format(epoch)
    trg_index = np.random.permutation(len(rules))[:20]
    trg_rules = []
    for batch in xrange(20):
        trg_rules.append(rules[trg_index[batch]])
    optim.zero_grads()
    loss = model.get_loss(trg_rules)
    loss.backward()
    optim.update()

if not os.path.isdir(dat_path):
    os.mkdir(dat_path)

serializers.save_hdf5(dat_path + "dqn.model", model)
serializers.save_hdf5(dat_path + "dqn.state", optim)

