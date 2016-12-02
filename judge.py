# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import pickle
import json

last_path = os.path.join(os.path.dirname(__file__), 'last/')

result_path = os.path.join(os.path.dirname(__file__), 'result/')

turn_path = os.path.join(os.path.dirname(__file__), 'turn/')

if not os.path.isfile(last_path + "0.pickle"):
    print "No such a file: " + last_path + "0.pickle"
    print "exit..."
    sys.exit()

with open(last_path + "0.pickle", "rb") as f:
    left = pickle.load(f)

if not os.path.isfile(last_path + "1.pickle"):
    print "No such a file: " + last_path + "1.pickle"
    print "exit..."
    sys.exit()

with open(last_path + "1.pickle", "rb") as f:
    right = pickle.load(f)

if not os.path.isfile(result_path + 'painted.txt'):
    print "No such a file: " + result_path + "painted.txt"
    print "exit..."
    sys.exit()

with open(result_path + 'painted.txt') as f:
    result = map(int, f.read().split(' '))

if result[0] > result[1]:
    print "left side win"
    with open(turn_path + "win.pickle", 'wb') as f:
        pickle.dump([left[0], left[1], 100 * result[0], np.zeros((17, 15), dtype=np.int32), True], f)
elif result[0] < result[1]:
    print "right side win"
    with open(turn_path + "win.pickle", 'wb') as f:
        pickle.dump([right[0], right[1], 100 * result[1], np.zeros((17, 15), dtype=np.int32), True], f)
else:
    print "draw"
    os.remove(turn_path + "win.pickle")

