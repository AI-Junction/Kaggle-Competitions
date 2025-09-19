# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 23:14:58 2017

@author: echtpar
"""

import multiprocessing 

def spawn():
    print('spawned!')
    
if __name__ == '__main__':
    for i in range(55):
        p = multiprocessing.Process(target=spawn)
        p.start()
        p.join()
    