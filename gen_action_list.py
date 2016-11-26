# -*- coding: utf-8 -*-

class ActionList:
    action_list = []

    action_cost = [1 << 28, 4, 4, 4, 4, 2, 2, 2, 2, 1]
    
    def __init__(self):
        self.dfs("0", 7)
        self.dfs("1", 7)
        self.dfs("2", 7)

    def dfs(self, action_str, power):
        act = action_str + " 0"
        self.action_list.append(act)
        
        for i in xrange(len(self.action_cost)):
            if self.action_cost[i] <= power:
                self.dfs(action_str + " " + str(i), power - self.action_cost[i])

    def get_action_size(self):
        return len(self.action_list)
    
    def get_action_str(self, index):
        return self.action_list[index]

    def get_action_idx(self, action_str):
        return self.action_list.index(action_str)
    
    def to_valid_actions(self, arr):
        power = 7
        dst = []
        for a in arr:
            if self.action_cost[a] <= power:
                dst.append(a)
                power -= self.action_cost[a]
        return dst

