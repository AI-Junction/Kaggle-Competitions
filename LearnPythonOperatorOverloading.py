# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:21:08 2017

@author: echtpar
"""

#import pygame
#import random
#from LearnPythonObjectOrientedBlob import Blob
#import numpy as np
#
#
#STARTING_BLUE_BLOBS = 10
#STARTING_RED_BLOBS = 3
#STARTING_GREEN_BLOBS = 5
#
#
#WIDTH = 800
#HEIGHT = 600
#WHITE = (255,255,255)
#BLUE = (0, 0, 255)
#RED = (255,0,0)
#
#game_display = pygame.display.set_mode((WIDTH, HEIGHT))
#pygame.display.set_caption("Blob, World")
#clock = pygame.time.Clock()


import pygame
import random
from LearnPythonObjectOrientedBlob import Blob
import numpy as np

STARTING_BLUE_BLOBS = 15
STARTING_RED_BLOBS = 15
STARTING_GREEN_BLOBS = 15

WIDTH = 800
HEIGHT = 600
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blob World")
clock = pygame.time.Clock()


#class BlueBlob(Blob):
#    
#    def __init__(self, x_boundary, y_boundary):
#        Blob.__init__(self, (0, 0, 255), x_boundary, y_boundary)
#
#    def __add__(self, other_blob):
#        if other_blob.color == (255, 0, 0):
#            self.size -= other_blob.size
#            other_blob.size -= self.size
#            
#        elif other_blob.color == (0, 255, 0):
#            self.size += other_blob.size
#            other_blob.size = 0
#            
#        elif other_blob.color == (0, 0, 255):
#            pass
#        else:
#            raise Exception('Tried to combine one or multiple blobs of unsupported colors!')
#            
#        
#class RedBlob(Blob):
#    def __init__(self, x_boundary, y_boundary):
#        Blob.__init__(self, (255, 0, 0), x_boundary, y_boundary)
#
#
#class GreenBlob(Blob):
#    def __init__(self, x_boundary, y_boundary):
#        Blob.__init__(self, (0, 255, 0), x_boundary, y_boundary)
#
#
#def is_touching(b1,b2):
#    return np.linalg.norm(np.array([b1.x,b1.y])-np.array([b2.x,b2.y])) < (b1.size + b2.size)


class BlueBlob(Blob):
    def __init__(self, x_boundary, y_boundary):
        Blob.__init__(self, (0,0,255), x_boundary, y_boundary)
    
    def __add__(self, other_blob):
        if other_blob.color == (255,0,0):
            self.size -= other_blob.size
            other_blob.size -= self.size
        elif other_blob.color == (0,255,0):
            self.size += other_blob.size
            other_blob.size = 0
        elif other_blob.color == (0,0,255):
            pass
        else:
            raise Exception ('Tried to combine one or multiple blobs of unsupported colors')
            

class RedBlob(Blob):
    def __init__(self, x_boundary, y_boundary):
        Blob.__init__(self, (255,0,0), x_boundary, y_boundary)
        
    
class GreenBlob(Blob):
    def __init__(self, x_boundary, y_boundary):
        Blob.__init__(self, (0,255,0), x_boundary, y_boundary)
        
def is_touching(b1, b2):
    return np.linalg.norm(np.array([b1.x, b1.y]) - np.array([b2.x, b2.y])) < (b1.size + b2.size)
        
def handle_collisions(blob_list):
    blues, greens, reds = blob_list
    for blue_id, blue_blob in blues.copy().items():
        for other_blobs in blues, greens, reds:
            for other_blob_id, other_blob in other_blobs.copy().items():
                if blue_blob == other_blob:
                    pass
                else:
                    if is_touching(blue_blob, other_blob):
                        blue_blob + other_blob
                        if other_blob.size <= 0:
                            del other_blobs[other_blob_id]
                        if blue_blob.size <= 0:
                            del blues[blue_id]
    return blues, greens, reds
    
#def handle_collisions(blob_list):
#    blues, reds, greens = blob_list
#    for blue_id, blue_blob in blues.copy().items():
#        for other_blobs in blues, reds, greens:
#            for other_blob_id, other_blob in other_blobs.copy().items():
#                if blue_blob == other_blob:
#                    pass
#                else:
#                    if is_touching(blue_blob, other_blob):
#                        blue_blob + other_blob
#                        if other_blob.size <= 0:
#                            del other_blobs[other_blob_id]
#                        if blue_blob.size <= 0:
#                            del blues[blue_id]
#                            
#    return blues, reds, greens

        
    
                      
#def draw_environment(blob_list):
#    game_display.fill(WHITE)
#    blues, reds, greens = handle_collisions(blob_list)
#    for blob_dict in blob_list:
#        for blob_id in blob_dict:
#            blob = blob_dict[blob_id]
#            pygame.draw.circle(game_display, blob.color, [blob.x, blob.y], blob.size)
#            blob.move()
##            blob.check_bounds()
#            if blob.x < 0: blob.x = 0
#            elif blob.x > blob.x_boundary: blob.x = blob.x_boundary
#            
#            if blob.y < 0: blob.y = 0
#            elif blob.y > blob.y_boundary: blob.y = blob.y_boundary
#
#    pygame.display.update()
#    return blues, reds, greens
                      
    

def draw_environment(blobs_list):
    game_display.fill(WHITE)
    blues, reds, greens = handle_collisions(blobs_list)
    for blob_dict in blobs_list:
        for blob_id in blob_dict:
            blob = blob_dict[blob_id]
            pygame.draw.circle(game_display, blob.color, [blob.x, blob.y], blob.size)
            blob.move()
#            blob.move_fast()
            if blob.x < 0: blob.x = 0
            elif blob.x > blob.x_boundary: blob.x = blob.x_boundary
            
            if blob.y < 0: blob.y = 0
            elif blob.y > blob.y_boundary: blob.y = blob.y_boundary
    pygame.display.update()
    return blues, reds, greens
    


#def main():
#    blue_blobs = dict(enumerate([BlueBlob(WIDTH,HEIGHT) for i in range(STARTING_BLUE_BLOBS)]))
#    red_blobs = dict(enumerate([RedBlob(WIDTH,HEIGHT) for i in range(STARTING_RED_BLOBS)]))
#    green_blobs = dict(enumerate([GreenBlob(WIDTH,HEIGHT) for i in range(STARTING_GREEN_BLOBS)]))
#
#    while True:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT:
#                pygame.quit()
#                quit()
#        blue_blobs, red_blobs, green_blobs = draw_environment([blue_blobs,red_blobs,green_blobs])
#        clock.tick(60)





def main():
#    red_blob = Blob(RED)
#    blue_blobs = dict(enumerate([BlueBlob(BLUE, WIDTH, HEIGHT, movement_range=(-3,3), size=(6,9)) for i in range(STARTING_BLUE_BLOBS)]))
    blue_blobs = dict(enumerate([BlueBlob(WIDTH, HEIGHT) for i in range(STARTING_BLUE_BLOBS)]))
    red_blobs = dict(enumerate([RedBlob(WIDTH, HEIGHT) for i in range(STARTING_RED_BLOBS)]))
    green_blobs = dict(enumerate([GreenBlob(WIDTH, HEIGHT) for i in range(STARTING_GREEN_BLOBS)]))
    
#    print('blue blob size {} and red blob size {}'.format(blue_blobs[0].size, red_blobs[0].size))
#    blue_blobs[0]+red_blobs[0]
#    print('blue blob size {} and red blob size {}'.format(blue_blobs[0].size, red_blobs[0].size))
    
    
#    print(blue_blobs)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        draw_environment([blue_blobs, red_blobs, green_blobs])
        clock.tick(60)
#        print(red_blob.x, red_blob.y)
        
if __name__ == '__main__'    :
    main()
        
         