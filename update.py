# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import optimizers, serializers
from dqn import DQN
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

model = DQN()
if os.path.isfile(dat_path + "dqn.model"):
    print >> sys.stderr, "::loading dqn.model..."
    serializers.load_hdf5(dat_path + "dqn.model", model)

optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in xrange(40):
    print "Epoch: {}".format(epoch)
    trg_index = np.random.permutation(len(rules))[:32]
    trg_rules = []
    for batch in xrange(32):
        trg_rules.append(rules[trg_index[batch]])
    optimizer.zero_grads()
    loss = model.get_loss(trg_rules)
    loss.backward()
    optimizer.update()

serializers.save_hdf5(dat_path + "dqn.model", model)

