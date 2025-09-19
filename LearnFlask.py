# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:13:23 2018

@author: echtpar
"""

from flask import Flask, request

app = Flask(__name__)

import cv2
import numpy as np
#import urllib3
import urllib.request
import urllib.error
import json
import argparse

import urllib
from urllib.request import urlopen
#response = urllib.urlopen('http://python.org/')
#html = response.read()


# import the necessary libs & functions
from flask import url_for


#@app.route('/', methods=["GET"])

#return 'Hello from AI- Junction2!'
#url = "https://www.openstack.org/themes/openstack/images/openstack-logo/OpenStack-Logo-Horizontal.png"
#req = urllib2.urlopen("https://www.openstack.org/themes/openstack/images/openstack-logo/OpenStack-Logo-Horizontal.png")
#img_url = request.args.get('url')
img_url = "https://i.imgur.com/4wgaVAt.jpg"
#return url
#try:
#con = urlopen("https://i.imgur.com/4wgaVAt.jpg")
#con = urlopen(img_url)
con = urllib.request.(img_url)
im_array = np.asarray(bytearray(con.read()), dtype = np.uint8)
im = cv2.imdecode(im_array, cv2.IMREAD_GRAYSCALE)
head, width = im.shape
print (json.dumps({"width": width, "height": head}))
#except urllib.error as e:
txt = ''
#print 'An error occured connecting to the wiki. No wiki page will be generated.'
print('<font color=\"red\">QWiki</font>')