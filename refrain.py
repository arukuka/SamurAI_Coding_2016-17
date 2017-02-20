# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import pickle
import json
import copy

from gen_action_list import ActionList

actions = ActionList()

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
        if action_str.split(' ')[1] == 0:
            return 0, False
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
            x, y = nx, ny
        if a == 9:
            # hide / show
            if not hide and field[y][x] in {3, 4, 5, 8}:
                return merit, True
            hide = not hide
    
    return merit, False


turn_path = os.path.join(os.path.dirname(__file__), 'turn/')

json_path = os.path.join(os.path.dirname(__file__), 'json/')

last_path = os.path.join(os.path.dirname(__file__), 'last/')

with open(json_path + "target.txt") as f:
    target = int(f.read())

print >> sys.stderr, target

with open(json_path + str(target) + ".json") as f:
    s = f.read()
    acts = json.loads(s)

if not os.path.isdir(turn_path):
    os.mkdir(turn_path)

if not os.path.isdir(last_path):
    os.mkdir(last_path)

side = int(input())

s0 = np.zeros((17, 15), dtype=np.int32)
s1 = np.zeros((17, 15), dtype=np.int32)
s2 = np.zeros((17, 15), dtype=np.int32)
s3 = np.zeros((17, 15), dtype=np.int32)

idx = side

HOME = [
    [0, 0],
    [0, 7],
    [7, 0],
    [14, 14],
    [14, 7],
    [7, 14]
]

enemies = []
for i in xrange(3):
    e = copy.copy(HOME[(1 - side) * 3 + i])
    enemies.append(e)

print 0
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
        enemies[0] = copy.copy(HOME[(1 - side) * 3 + 0])
    if s0[16][9] > 0:
        enemies[1] = copy.copy(HOME[(1 - side) * 3 + 1])
    if s0[16][14] > 0:
        enemies[2] = copy.copy(HOME[(1 - side) * 3 + 2])
    state = [s0, s1, s2, s3]
    if idx - 1 >= 0:
        jdx = idx - 1
        e = int(acts['plays'][jdx]['samurai'])
        x = enemies[e][0]
        y = enemies[e][1]
        for act in map(int, acts['plays'][jdx]['actions']):
            if 5 <= act and act <= 8:
                # move
                dx, dy = rotate(0, 1, act - 5)
                x = x + dx
                y = y + dy
        enemies[e][0] = x
        enemies[e][1] = y
            
    kumi = [
        state,
        enemies[0][0] + enemies[0][1] * 15,
        enemies[1][0] + enemies[1][1] * 15,
        enemies[2][0] + enemies[2][1] * 15
    ]
    print >> sys.stderr, "enemies: {}".format(enemies)
    with open(turn_path + str(turn) + '.pickle', mode='wb') as f:
        pickle.dump(kumi, f)
    action_str = " ".join(([str(acts['plays'][idx]['samurai'])] + map(str, actions.to_valid_actions(acts['plays'][idx]['actions']))) + ["0"])
    action = actions.get_action_idx(action_str)
    s3 = s2.copy()
    s2 = s1.copy()
    s1 = s0.copy()
    print acts['plays'][idx]['samurai']
    print " ".join(map(str, acts['plays'][idx]['actions']))
    print 0
    sys.stdout.flush()
    idx = idx + 2

