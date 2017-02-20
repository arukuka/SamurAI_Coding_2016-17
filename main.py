# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import pickle

from chainer import serializers

from dqn import DQN

dat_path = os.path.join(os.path.dirname(__file__), 'dat/')

s0 = np.zeros((17, 15), dtype=np.int32)
s1 = np.zeros((17, 15), dtype=np.int32)
s2 = np.zeros((17, 15), dtype=np.int32)
s3 = np.zeros((17, 15), dtype=np.int32)

dqn = DQN()

for i in xrange(3):
    if os.path.isfile(dat_path + "dqn" + str(i) + ".model"):
        print >> sys.stderr, "::loading dqn" + str(i) + ".model..."
        serializers.load_hdf5(dat_path + "dqn" + str(i) + ".model", dqn.models[i])

side = input()
print 0

def write(p, P):
    if P > 0.8:
        print "{} {}".format(p % 15, p / 15)
    else:
        print "-1 -1"


while True:
    turn = input()
    for i in xrange(3):
        a = map(int, raw_input().split())
        for j in xrange(5):
            s0[15][i * 5 + j] = a[j]
    for i in xrange(3):
        a = map(int, raw_input().split())
        for j in xrange(5):
            s0[16][i * 5 + j] = a[j]
    for i in xrange(15):
        a = map(int, raw_input().split())
        for j in xrange(15):
            s0[i][j] = a[j]
    state = [s0, s1, s2, s3]
    p0, P0, p1, P1, p2, P2 = dqn.next_action(state)
    write(p0, P0)
    write(p1, P1)
    write(p2, P2)
    sys.stdout.flush()
    s3 = s2.copy()
    s2 = s1.copy()
    s1 = s0.copy()

