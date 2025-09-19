# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:21:08 2017

@author: echtpar
"""

import random


class Blob:
    def __init__(self, color, x_boundary, y_boundary, size = (4,8), movement_range=(-1,2)):
        self.color = color
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.x = random.randrange(0, self.x_boundary)
        self.y = random.randrange(0, self.y_boundary)
        self.size = random.randrange(size[0],size[1])
        self.movement_range = movement_range
        
        
    def move(self):
        self.move_x = random.randrange(self.movement_range[0], self.movement_range[1])
        self.move_y = random.randrange(self.movement_range[0], self.movement_range[1])
        self.x += self.move_x
        self.y += self.move_y
        


