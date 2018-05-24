#coding:utf-8
#@author:zee(GDUT)
# 工具：pycharm
import pandas as pd
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# import random
# new_data=pd.read_csv('D:/test01/features.dat')
# sep =int(0.7*len(new_data))
train_data = pd.read_csv('D:/test01/features.dat')
test_data = pd.read_csv('D:/test01/labels.dat')
# 开始搭建神经网络
tf_input = tf.placeholder(tf.float32,[None,25],"input")
tfx=tf_input[:,:21]
tfy=tf_input[:,21:]
L1=tf.layers.dense(tfx,128,tf.nn.relu,name="L1")
L2=tf.layers.dense(L1,128,tf.nn.relu,name="L2")
out=tf.layers.dense(L2,4,name="output")
prediction=tf.nn.softmax(out,name="pred")

loss=tf.losses.softmax_cross_entropy(onehot_labels=tfy,logits=out)
accuracy=tf.metrics.accuracy(
labels=tf.argmax(tfy,axis=1),
predictions=tf.argmax(out,axis=1))[1]
opt=tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op=opt.minimize(loss)
sess=tf.Session()
sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
#plt.ion()
#fig,(ax1,ax2)=plt.subplots(1,2,figsize(8,4))
#开始训练网络
for t in range(6000):
    batch_index=np.random.randint(len(train_data),size=32)
    sess.run(train_op,{tf_input:train_data[batch_index]})
    if t % 50 ==0:
        acc_,pre_,loss_=sess.run([accuracy,prediction,loss],{tf_input:test_data})
        print("step:%i"%t,"|Accuate:%.3f"%acc_,"|Loss:%.3f"%loss_)






