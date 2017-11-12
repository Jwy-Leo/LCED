# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from flag import *


BN_EPSILON = 0.001


def variable_summaries( name, var):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

        #tf.contrib.layers.xavier_initializer()
def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        if is_fc_layer is True:
            regularizer = tf.contrib.layers.l2_regularizer(scale=flags.weight_decay)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=flags.weight_decay)


        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables


def output_layer(input_layer, num_labels,num):

        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = create_variables(name='fc_weights'+str(num), shape=[input_dim, num_labels], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = create_variables(name='fc_bias'+str(num), shape=[num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.nn.bias_add(tf.matmul(input_layer, fc_w), fc_b)
        return fc_h


def batch_normalization_layer(input_layer, dimension):
        x_shape = input_layer.get_shape()
        params_shape = x_shape[-1:]
        print(params_shape)

        moving_mean = tf.get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.constant_initializer(0.0, tf.float32),
                                    trainable=False)

        moving_variance = tf.get_variable('moving_variance',
                                        params_shape, tf.float32,
                                          initializer=tf.constant_initializer(1.0, tf.float32),
                                        trainable=False)

        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        av=0.99999
        moving_mean = moving_averages.assign_moving_average(moving_mean,mean, av)

        moving_variance = moving_averages.assign_moving_average(moving_variance, variance,av)

        tf.add_to_collection('resnet_update_ops', moving_mean)
        tf.add_to_collection('resnet_update_ops', moving_variance)

        beta = tf.get_variable('beta', dimension, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        '''
        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))
        '''
        if flags.train:
            mean, variance =(mean, variance)
        else:
            print('test')
            mean, variance =(moving_mean, moving_variance)


        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

        return bn_layer


def conv_bn_relu_layer(name,input_layer, filter_shape, stride):

        out_channel = filter_shape[-1]
        filter = create_variables(name, shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = batch_normalization_layer(conv_layer, out_channel)

        output = tf.nn.relu(bn_layer)
        return output

def bn_relu_conv_layer(name,input_layer, filter_shape, stride):

        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)

        filter = create_variables(name, shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer



def residual_block(name,input_layer, output_channel, first_block=False):

        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = create_variables(name+"_first", shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = bn_relu_conv_layer(name+"_first",input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = bn_relu_conv_layer(name+"_second",conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output


def inference(input_tensor_batch, n,reuse=False):
        layers = []
        with tf.variable_scope('conv0'):
            conv0 = conv_bn_relu_layer('conv0',input_tensor_batch, [3, 3, 1, 16], 1)
            #activation_summary(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' %i):
                if i == 0:
                    conv1 = residual_block('conv1_%d' %i,layers[-1], 16, first_block=True)
                else:
                    conv1 = residual_block('conv1_%d' %i,layers[-1], 16)
               # activation_summary(conv1)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' %i):
                conv2 = residual_block('conv2_%d' %i,layers[-1], 32)
                #activation_summary(conv2)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' %i):
                conv3 = residual_block('conv3_%d' %i,layers[-1], 64)
                layers.append(conv3)

           # assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            #assert global_pool.get_shape().as_list()[-1:] == [64]
            #output = output_layer(global_pool, 100)
            layers.append(global_pool)

        return layers[-1]



