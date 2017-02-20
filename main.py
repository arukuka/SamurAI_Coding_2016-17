# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import pickle

from chainer import serializers

from dqn import DQN

weaponSections = [ 4, 5, 7 ]
weaponReach = [
  [0, 1, 0, 2, 0, 3, 0, 4],
  [0, 1, 0, 2, 1, 0, 1, 1, 2, 0],
  [-1, -1, -1, 0, -1, 1, 0, 1, 1, -1, 1, 0, 1, 1]
]

def rotate(dx, dy, direction):
    if direction == 0:
        return dx, dy
    elif direction == 1:
        return dy, -dx
    elif direction == 2:
        return -dx, -dy
    elif direction == 3:
        return -dy, dx
    else:
        print >> sys.stderr, "invalid direction in rotate"
    raise Exception("invalid direction in rotate")

def simulate(field, action_str):
    merit = 0
    
    samuraiID = int(action_str.split(' ')[0])
    x = field[15][samuraiID * 5]
    y = field[15][samuraiID * 5 + 1]
    done = field[15][samuraiID * 5 + 2] == 1
    hide = field[15][samuraiID * 5 + 3] == 1
    rest = field[15][samuraiID * 5 + 4]
    
    if done or rest > 0:
        return 0, True
    
    for a in map(int, action_str.split(' ')[1:-1]):
        if 1 <= a and a <= 4:
            # attack
            if hide:
                return merit, True
            for i in xrange(weaponSections[samuraiID]):
                dx, dy = rotate(weaponReach[samuraiID][2 * i], weaponReach[samuraiID][2 * i + 1], a - 1)
                nx = x + dx
                ny = y + dy
                if nx < 0 or 15 <= nx or ny < 0 or 15 <= ny:
                    continue
                if (nx, ny) in {(0, 0), (0, 7), (7, 0), (14, 14), (14, 7), (7, 14)}:
                    continue
                if field[ny][nx] == 8:
                    merit += 25
                elif 3 <= field[ny][nx] and field[ny][nx] <= 5:
                    merit += 20
                field[ny][nx] = samuraiID
        if 5 <= a and a <= 8:
            # move
            dx, dy = rotate(0, 1, a - 5)
            nx = x + dx
            ny = y + dy
            if nx < 0 or 15 <= nx or ny < 0 or 15 <= ny:
                return merit, True
            if hide and field[ny][nx] in {3, 4, 5, 8}:
                return merit, True
            print >> sys.stderr, "move from {} to {}".format([x, y], [nx, ny])
            x, y = nx, ny
        if a == 9:
            # hide / show
            if not hide and field[y][x] in {3, 4, 5, 8}:
                return merit, True
            hide = not hide
    
    return merit, False

turn_path = os.path.join(os.path.dirname(__file__), 'turn/')

temp_path = os.path.join(os.path.dirname(__file__), 'temp/')

dat_path = os.path.join(os.path.dirname(__file__), 'dat/')

last_path = os.path.join(os.path.dirname(__file__), 'last/')

side = int(input())

s0 = np.zeros((17, 15), dtype=np.int32)
s1 = np.zeros((17, 15), dtype=np.int32)
s2 = np.zeros((17, 15), dtype=np.int32)
s3 = np.zeros((17, 15), dtype=np.int32)
reward = 0
action = 0
prev_state = [s0.copy(), s1, s2, s3]

dqn = DQN()
if os.path.isfile(dat_path + "dqn.model"):
    print >> sys.stderr, "::loading dqn.model..."
    serializers.load_hdf5(dat_path + "dqn.model", dqn.model)

temp = 0.0
if os.path.isfile(temp_path + "temp.pickle"):
    with open(temp_path + "temp.pickle", "rb") as f:
        temp = pickle.load(f)

if not os.path.isdir(temp_path):
    os.mkdir(temp_path)

if not os.path.isdir(turn_path):
    os.mkdir(turn_path)

if not os.path.isdir(last_path):
    os.mkdir(last_path)

with open(temp_path + 'temp.pickle', mode='wb') as f:
    pickle.dump(temp + 0.5, f)

def get_prob():
    return np.exp(-(temp*1.5174271293851464/27000)**2)

print "0"
sys.stdout.flush()

while True:
    turn = int(input())
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
    if s0[16][4] > 0:
        reward += 1000
    if s0[16][9] > 0:
        reward += 1000
    if s0[16][14] > 0:
        reward += 1000
    state = [s0, s1, s2, s3]
    kumi = [prev_state, action, reward, state, False]
    with open(turn_path + str(turn) + '.pickle', mode='wb') as f:
        pickle.dump(kumi, f)
    action, action_str = dqn.next_action(state, 0)
    merit, invalid_flag = simulate(s0.copy(), action_str)
    reward = merit
    prev_state =  [s0.copy(), s1.copy(), s2.copy(), s3.copy()]
    s3 = s2.copy()
    s2 = s1.copy()
    s1 = s0.copy()
    if turn + 2 >= 96:
        with open(last_path + str(side) + '.pickle', mode='wb') as f:
            pickle.dump([prev_state, action], f)
    print action_str
    sys.stdout.flush()

