# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 19:57:49 2017

@author: echtpar
"""

import numpy as np
import matplotlib.pyplot as plt


def classval(x):
    if x>0.5: return 1 else: return 0

z = lambda x: 1 if x > 0.5 else 0
    

print(z(0.25))


llimit1 = np.percentile(p_proba[:,(0)], (100-30))
llimit2 = np.percentile(p_proba[:,(1)], (100-0.8))
llimit3 = np.percentile(p_proba[:,(2)], (100-2.1))
llimit4 = np.percentile(p_proba[:,(3)], (100-0.8))
llimit5 = np.percentile(p_proba[:,(4)], (100-0.2))
llimit6 = np.percentile(p_proba[:,(5)], (100-30))
llimit7 = np.percentile(p_proba[:,(6)], (100-5))
llimit8 = np.percentile(p_proba[:,(7)], (100-0.2))
llimit9 = np.percentile(p_proba[:,(8)], (100-11))
llimit10 = np.percentile(p_proba[:,(9)], (100-9))
llimit11 = np.percentile(p_proba[:,(10)], (100-6))
llimit12 = np.percentile(p_proba[:,(11)], (100-18))
llimit13 = np.percentile(p_proba[:,(12)], (100-93))
llimit14 = np.percentile(p_proba[:,(13)], (100-20))
llimit15 = np.percentile(p_proba[:,(14)], (100-0.8))
llimit16 = np.percentile(p_proba[:,(15)], (100-0.5))
llimit17 = np.percentile(p_proba[:,(16)], (100-18))

#llimit = np.percentile(p_proba[:][19], 50)

print(llimit1, llimit2, llimit3, llimit4, llimit5, llimit6, llimit7, llimit8, llimit9, llimit10, llimit11, llimit12, llimit13, llimit14, llimit15, llimit16, llimit17)

p_proba_class = pd.DataFrame(p_proba)
p_proba_class.head(1)

p_proba_class['agriculture'] = p_proba_class['agriculture'].apply (lambda x: 1 if x >= llimit1 else 0)
p_proba_class['artisinal_mine'] = p_proba_class['artisinal_mine'].apply (lambda x: 1 if x >= llimit2 else 0)
p_proba_class['bare_ground'] = p_proba_class['bare_ground'].apply (lambda x: 1 if x >= llimit3 else 0)
p_proba_class['blooming'] = p_proba_class['blooming'].apply (lambda x: 1 if x >= llimit4 else 0)
p_proba_class['blow_down'] = p_proba_class['blow_down'].apply (lambda x: 1 if x >= llimit5 else 0)
p_proba_class['clear'] = p_proba_class['clear'].apply (lambda x: 1 if x >= llimit6 else 0)
p_proba_class['cloudy'] = p_proba_class['cloudy'].apply (lambda x: 1 if x >= llimit7 else 0)
p_proba_class['conventional_mine'] = p_proba_class['conventional_mine'].apply (lambda x: 1 if x >= llimit8 else 0)
p_proba_class['cultivation'] = p_proba_class['cultivation'].apply (lambda x: 1 if x >= llimit9 else 0)
p_proba_class['habitation'] = p_proba_class['habitation'].apply (lambda x: 1 if x >= llimit10 else 0)
p_proba_class['haze'] = p_proba_class['haze'].apply (lambda x: 1 if x >= llimit11 else 0)
p_proba_class['partly_cloudy'] = p_proba_class['partly_cloudy'].apply (lambda x: 1 if x >= llimit12 else 0)
p_proba_class['primary'] = p_proba_class['primary'].apply (lambda x: 1 if x >= llimit13 else 0)
p_proba_class['road'] = p_proba_class['road'].apply (lambda x: 1 if x >= llimit14 else 0)
p_proba_class['selective_logging'] = p_proba_class['selective_logging'].apply (lambda x: 1 if x >= llimit15 else 0)
p_proba_class['slash_burn'] = p_proba_class['slash_burn'].apply (lambda x: 1 if x >= llimit16 else 0)
p_proba_class['water'] = p_proba_class['water'].apply (lambda x: 1 if x >= llimit17 else 0)