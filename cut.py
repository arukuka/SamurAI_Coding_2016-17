# -*- coding: utf-8 -*-

import os.path
import sys
import os
import pickle
import math

turn_path = os.path.join(os.path.dirname(__file__), 'turn/')

side = int(sys.argv[1])

files = os.listdir(turn_path)
rules = []
for fn in files:
    t = int(fn.split(".")[0])
    if t % 2 == side:
        continue
    os.remove(turn_path + fn)

