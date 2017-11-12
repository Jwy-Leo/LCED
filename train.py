from sklearn.model_selection import train_test_split
import h5py
import cv2
import matplotlib.pyplot as plt
from resnet import *
import tensorflow as tf
import numpy as np
from flag import *



def concat_channels(r, g, b):
    assert r.ndim == 2 and g.ndim == 2 and b.ndim == 2
    rgb = (r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis])
    return np.concatenate(rgb, axis=-1)

def plotim(im, i, printcolor=True, ):
    ax = plt.figure(i).add_subplot(1, 1, 1)
    if printcolor:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    else:
        if len(im.shape) >= 3:
            plot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')

        else:
            ax.imshow(im, cmap='gray')
def shuffle_data(data, labels):
    data, _, labels, _ = train_test_split(
        data, labels, test_size=0.0, random_state=900
    )
    return data, labels
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




with h5py.File('train/train.h5', 'r') as f:
    train_R = f['R'][()][...,np.newaxis]
    print("read train_R shape as:" + str(train_R.shape))
    train_G = f['G'][()][...,np.newaxis]
    print("read train_G shape as:" + str(train_G.shape))
    train_B = f['B'][()][...,np.newaxis]
    print("read train_B shape as:" + str(train_B.shape))
    train_D = f['label_D'][()]
    print("read train_D shape as:" + str(train_D.shape))
    train_class = f['label'][()]
    print("read train_class shape as:" + str(train_class.shape))
    train_W = tranW(f['label_W'][()])
    print("read train_W shape as:" + str(train_W.shape))


def top_k_error( predictions, labels, k):

        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)



for i in range(0):
    plotim(train_R[i,:,:,0],1,False)
    plotim(train_G[i,:,:,0],2,False)
    plotim(train_B[i,:,:,0],3,False)
    print(np.argmax(train_class[i]))
    plt.pause(100)


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
with tf.variable_scope("loss"):
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=setout, labels=label, name='cross_entropy_per_example')
    predictions = tf.nn.softmax(setout)
    #cross_entropy = -tf.reduce_sum(label * tf.log(tf.clip_by_value(predictions,1e-10,1)))
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    full_loss = loss +0.01* sum(regu_losses)

#######################################train##############################
#train = tf.train.AdamOptimizer(learning_rate=lr).minimize(full_loss)
train = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9).minimize(full_loss)


################################prediction################################
with tf.variable_scope("prediction"):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





################################show################################
tf.summary.scalar('learning_rate', lr)
tf.summary.scalar('train_loss', full_loss )
tf.summary.scalar('train_accuracy', accuracy)


train_merged = tf.summary.merge_all()






fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.title.set_text('test_loss')
ax2=fig.add_subplot(2,1,2)
ax2.title.set_text('test_top1_Accuary')
ax2.set_ylim([0, 1])
plt.ion()
plt.show()
last_time_loss = 0.0
last_time_acc = 0.0

with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'logs/model10000.ckpt')
        print 'Restored from checkpoint...'
        #sess.run(tf.global_variables_initializer())
        summary_writer_train = tf.summary.FileWriter('logs/train', sess.graph)
        summary_writer_test = tf.summary.FileWriter('logs/test', sess.graph)
        train_account =train_R.shape[0]
        test_account =test_R.shape[0]
        batch=flags.train_batch_size
        tbatch=flags.test_batch_size

        l=flags.init_lr
        for i in range(10001,flags.train_maxiter+1):

            if i==flags.decay_step0 or i==flags.decay_step1:
                l=l*flags.lr_decay_factor

            if i%flags.report_freq==0:

                tnum = (tbatch * i) % (test_account - tbatch)
                [summary,lossout,acc] = sess.run([train_merged,full_loss,accuracy], feed_dict={
                                                    R: test_R[tnum:tnum + tbatch],
                                                    G: test_G[tnum:tnum + tbatch],
                                                    B: test_B[tnum:tnum + tbatch],
                                                    D: test_D[tnum:tnum + tbatch],
                                                    label: test_class[tnum:tnum + tbatch],
                                                    label_w: test_W[tnum:tnum + tbatch],

                                                    lr: 0.0})
                if i/flags.report_freq!=0:
                    ax.plot([i / flags.report_freq, i / flags.report_freq + 1], [last_time_loss, lossout], 'r-')
                    ax2.plot([i / flags.report_freq, i / flags.report_freq + 1], [last_time_acc, acc], 'r-')
                    plt.pause(1)
                print 'Interation = ', i
                print 'L = ', l
                print 'Validation top1 error = %.4f' % acc
                print 'Validation loss = ', lossout
                print '----------------------------'
                last_time_loss=lossout
                last_time_acc = acc
                summary_writer_test.add_summary(summary, i)

            num = (batch * i) % (train_account - batch)
            [a,summary]=sess.run([ train,train_merged ], feed_dict={
                                                        R: train_R[num:num + batch],
                                                        G: train_G[num:num + batch],
                                                        B: train_B[num:num + batch],
                                                        D: train_D[num:num + batch],
                                                        label: train_class[num:num + batch],
                                                        label_w: train_W[num:num + batch],
                                                        lr: l})
            summary_writer_train.add_summary(summary, i)
            if i % 3000 == 0 and i  != 0:
                save_path = saver.save(sess, "logs/model"+str(i)+".ckpt")
                print("Model saved in file: %s" % save_path)
            #print(n.shape)





