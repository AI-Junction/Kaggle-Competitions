# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:43:34 2017

@author: echtpar
"""

"""

TENSORFLOW START

"""


import tensorflow as tf
import tensorflow.tensorboard as tb
import numpy as np

x1 = tf.constant(5)
x2 = tf.constant(6)

result = x1 * x2
print(result)


result = tf.multiply(x1,x2)
print(result)

sess = tf.Session()
print(sess.run(result))


with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output) 
writer = tf.summary.FileWriter('C://tmp//tensorflow_logs', graph=sess.graph)
tensorboard --logdir="C://tmp//tensorflow_logs"
sess.close()



a = tf.constant(3)
sess = tf.Session()
with sess.as_default():
    print(a.eval())

sess.close()    

#tf.Session.__init__(target = "", graph = None, config = None)
tf.Session.__init__()

a = tf.constant([10,20])
b = tf.constant([1,2])
sess = tf.Session()
v = sess.run(a)
print(v)
v = sess.run([a,b])
print(v)
with sess.as_default():
    print(b.eval())



with tf.Session() as sess:
    v = sess.run(a)
    print(v)
    v = sess.run([a,b])
    print(v)


a = tf.add(2,3)
b = tf.multiply(a,5)
sess = tf.Session()
replace_dict = {a:15}
print(sess.run(b, replace_dict))

x = tf.placeholder(tf.float32, shape = (1024,1024))
y = tf.matmul(x,x)

sess = tf.Session()

with tf.Session() as sess:
    rand_dict = np.random.rand(1024,1024)
    output = sess.run(y, feed_dict={x:rand_dict})
    print(output.shape)

print(rand_dict)    


d = tf.placeholder(tf.int32, shape = [2], name = "input_d")
e = tf.reduce_prod(d, name="product_e")
f = tf.reduce_sum(d, name="sum_f")
g = tf.add(e,f, name="add_g")
sess = tf.Session()
input_dict = {d:np.array([2,3], dtype = np.int32)}
print(sess.run(g, feed_dict = input_dict))



my_var = tf.Variable(4, name="my_var")
add = tf.add(5, my_var)
prod = tf.multiply(4, my_var)

sess = tf.Session()
sess.run(my_var.initializer)
print(sess.run(my_var))
print(sess.run(add))
print(sess.run(prod))

print(my_var.eval(session = sess))
print(add.eval(session = sess))
print(prod.eval(session = sess))


ones = tf.ones([4,4])
zeroes = tf.zeros([3,2])
random_uniform = tf.random_uniform([3,4], minval=0, maxval = 10)

sess = tf.Session()
print(sess.run(zeroes))
print(sess.run(ones))
print(sess.run(random_uniform))

var = tf.Variable(zeroes)
sess.run(var.initializer)
print(var.eval(session = sess))

var = tf.Variable(tf.zeros([2,3]))
sess.run(var.initializer)
print(var.eval(session = sess))


myvar1 = tf.Variable(4)
myvar2 = tf.Variable(5)

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
print(myvar1.eval(session = sess))
print(myvar2.eval(session = sess))

myvar = tf.Variable(0)
increment_by_two = myvar.assign(myvar + 2)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(myvar.eval(session = sess))
sess.run(increment_by_two)
print(myvar.eval(session = sess))


import tensorflow as tf
myvar = tf.Variable(0)
init = tf.initialize_all_variables()
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(init)
sess1.run(myvar.assign_add(1))
print(myvar.eval(session=sess1))

sess2.run(init)
sess2.run(myvar.assign_add(2))
print(myvar.eval(session=sess2))

sess1.run(myvar.assign_add(5))
print(myvar.eval(session=sess1))

sess2.run(myvar.assign_add(2))
print(myvar.eval(session=sess2))


import tensorflow as tf

with tf.name_scope("Scope1"):
    a=tf.add(1,2, name="Scope1_add")
    b=tf.multiply(a,3,name="Scope1_mul")
with tf.name_scope("Scope2"):
    c=tf.add(4,5,name="Scope2_add")
    d=tf.multiply(c,6,name="Scope2_mul")
e=tf.add(b,d,name="output")    

writer = tf.summary.FileWriter("C://name_scope", graph=tf.Graph)    
writer.close()



