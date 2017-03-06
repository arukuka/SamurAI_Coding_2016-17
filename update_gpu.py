# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import optimizers, serializers, cuda
from dqn_gpu import DQN
import os.path
import sys
import os
import pickle
import math

turn_path = os.path.join(os.path.dirname(__file__), 'turn/')

files = os.listdir(turn_path)
rules = []
for fn in files:
    with open(turn_path + fn, "rb") as f:
        r = pickle.load(f)
        rules.append(r)
    os.remove(turn_path + fn)

dat_path = os.path.join(os.path.dirname(__file__), 'dat/')

cuda.get_device(0).use()
dqn = DQN()

optims = []

for i in xrange(3):
    optim = optimizers.Adam(alpha = 0.0001)
    optim.setup(dqn.models[i])

    if os.path.isfile(dat_path + "dqn" + str(i) + ".model"):
        print >> sys.stderr, "::loading dqn.model..."
        serializers.load_npz(dat_path + "dqn" + str(i) + ".model", dqn.models[i])

    if os.path.isfile(dat_path + "dqn" + str(i) + ".state"):
        print >> sys.stderr, "::loading dqn.state..."
        serializers.load_npz(dat_path + "dqn" + str(i) + ".state", optim)

    dqn.models[i] = dqn.models[i].to_gpu()
    optims.append(optim)

for epoch in xrange(100):
    print "Epoch: {}".format(epoch)
    num = min(len(rules), 20)
    trg_index = np.random.permutation(len(rules))[:num]
    trg_rules = []
    for batch in xrange(num):
        trg_rules.append(rules[trg_index[batch]])
    for optim in optims:
        optim.zero_grads()
    loss0, loss1, loss2 = dqn.get_loss(trg_rules)
    loss0.backward()
    loss1.backward()
    loss2.backward()
    for optim in optims:
        optim.update()
    if loss0.data == 0:
        print "\tloss0 : ZERO"
    else:
        print "\tloss0 : {}".format(math.log10(float(loss0.data)))
    if loss1.data == 0:
        print "\tloss1 : ZERO"
    else:
        print "\tloss1 : {}".format(math.log10(float(loss1.data)))
    if loss2.data == 0:
        print "\tloss2 : ZERO"
    else:
        print "\tloss2 : {}".format(math.log10(float(loss2.data)))

if not os.path.isdir(dat_path):
    os.mkdir(dat_path)

for i in xrange(3):
    serializers.save_npz(dat_path + "dqn" + str(i) + ".model", dqn.models[i].to_cpu())
    serializers.save_npz(dat_path + "dqn" + str(i) + ".state", optims[i])

