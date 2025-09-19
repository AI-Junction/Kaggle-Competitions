# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 05:58:52 2017

@author: echtpar
"""

from multiprocessing import Pool

def job(num):
    return num*2
    
if __name__ == '__main__'    :
    p = Pool(processes = 20)
    data = p.map(job, range(20))
    data2 = p.map(job, [2,3,4,5])
    
    p.close()
    print(data)
    print(data2)
    
                                                                                