from sklearn.model_selection import train_test_split
import h5py
import cv2
import matplotlib.pyplot as plt
from resnet import *
import tensorflow as tf
import numpy as np
from flag import *




def plotim(im, i, printcolor=True, ):
    ax = plt.figure(i).add_subplot(1, 1, 1)
    if printcolor:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    else:
        if len(im.shape) >= 3:
            plot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')

        else:
            ax.imshow(im, cmap='gray')

def tranW(label):
    size=label.shape
    out=np.zeros([size[0],2])
    for i in range(size[0]):
        out[i][label[i].astype('int32')]=1

    return out


with h5py.File('test/test.h5', 'r') as f:
    test_R= f['R'][()][...,np.newaxis]
    print("read test_R shape as:" + str(test_R.shape))
    test_G = f['G'][()][...,np.newaxis]
    print("read test_G shape as:" + str(test_G.shape))
    test_B = f['B'][()][...,np.newaxis]
    print("read test_B shape as:"+str(test_B.shape))
    test_D = f['label_D'][()]
    print("read test_D shape as:" + str(test_D.shape))
    test_class = f['label'][()]
    print("read test_class shape as:" + str(test_class.shape))
    test_W = tranW(f['label_W'][()])
    print("read test_W shape as:" + str(test_W.shape))




'''
for i in range(len(test_R)):
    plotim(test_R[i,:,:,0],1,False)
    plotim(test_G[i,:,:,0],2,False)
    plotim(test_B[i,:,:,0],3,False)
    print(np.argmax(test_class[i]))
    plt.pause(0.5)
'''


################################input################################
R= tf.placeholder(dtype=tf.float32,shape=[None,100,60, 1])
G= tf.placeholder(dtype=tf.float32,shape=[None,100,60, 1])
B= tf.placeholder(dtype=tf.float32,shape=[None,100,60, 1])
D= tf.placeholder(dtype=tf.float32, shape=[None,1])
label= tf.placeholder(dtype=tf.float32, shape=[None,10])
label_w= tf.placeholder(dtype=tf.float32, shape=[None,2])
lr= tf.placeholder(tf.float32)

################################resnet################################
with tf.variable_scope("resnet") as scope:
    logits_R = inference(R, flags.num_residual_blocks)
    scope.reuse_variables()
    logits_G = inference(G, flags.num_residual_blocks)
    scope.reuse_variables()
    logits_B = inference(B, flags.num_residual_blocks)

with tf.variable_scope("fc"):

    set_0=output_layer(logits_R,10,0)
    set_1=output_layer(logits_G,10,1)
    set_2=output_layer(logits_B, 10, 2)
    set = tf.concat([set_0, set_1, set_2, D], axis=1)
    #set=tf.nn.relu(set)
    setout=output_layer(set,10,3)

################################train_loss################################
predictions = tf.nn.softmax(setout)
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:
    saver = tf.train.Saver()
    if True:
        saver.restore(sess, 'logs/model21000.ckpt')
        print 'Restored from checkpoint...'
       # sess.run(tf.global_variables_initializer())
        accmatrix=np.zeros([10,10])
        accall=0.0


        for i in range(len(test_R)/flags.train_batch_size):
            print(i)
            [acc,prei] = sess.run([accuracy,predictions], feed_dict={
                                                R: test_R[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size],
                                                G: test_G[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size],
                                                B: test_B[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size],
                                                D: test_D[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size],
                                                label: test_class[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size],
                                                label_w: test_W[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size],
                                                lr: 0.0})

            pre=np.argmax(prei,axis=1)
            label_ = np.argmax(test_class[i*flags.train_batch_size:i*flags.train_batch_size+flags.train_batch_size], axis=1)
            print prei.shape
            print label_.shape
            print pre
            print label_
            accall = accall + acc
            print accall / (i + 1)
        '''
        for i in range(len(test_R) / flags.train_batch_size):
            print(i)
            [acc, prei] = sess.run([accuracy, predictions], feed_dict={
                    R: test_R[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size],
                    G: test_G[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size],
                    B: test_B[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size],
                    D: test_D[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size],
                    label: test_class[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size],
                    label_w: test_W[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size],
                    lr: 0.0})


            pre = np.argmax(prei, axis=1)
            label_ = np.argmax(test_class[i * flags.train_batch_size:i * flags.train_batch_size + flags.train_batch_size], axis=1)
            print prei.shape
            print label_.shape
            print pre
            print label_
            accall=accall+acc
            print accall/(i+1)
            #accmatrix[label_,pre[0]]=accmatrix[label_,pre[0]]+1

        '''







