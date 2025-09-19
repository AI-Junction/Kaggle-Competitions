# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 23:07:35 2018

@author: echtpar
"""

"""Filename: hello-world.py
  """

from flask import Flask

app = Flask(__name__)

@app.route('/users/<string:username>')
def hello_world(username=None):
    return("Hello {}!".format(username))