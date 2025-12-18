# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    """
    手写版 BatchNorm，不依赖 tf.layers / keras。
    训练时使用 batch mean/var 并更新 moving_mean / moving_var；
    测试时使用 moving_mean / moving_var。
    同时兼容 is_training 是 Python bool 或 tf.bool 张量两种情况。
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 通道维度
        params_shape = x.get_shape()[-1:]

        beta = tf.get_variable(
            'beta', params_shape,
            initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(
            'gamma', params_shape,
            initializer=tf.constant_initializer(1.0))

        moving_mean = tf.get_variable(
            'moving_mean', params_shape,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_var = tf.get_variable(
            'moving_var', params_shape,
            initializer=tf.ones_initializer(),
            trainable=False)

        # 除通道外的所有维度
        axes = list(range(len(x.get_shape()) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')

        def train_bn():
            # 更新滑动平均
            update_mean = tf.assign(
                moving_mean, moving_mean * decay + batch_mean * (1.0 - decay))
            update_var = tf.assign(
                moving_var, moving_var * decay + batch_var * (1.0 - decay))
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(
                    x, batch_mean, batch_var, beta, gamma, epsilon)

        def infer_bn():
            return tf.nn.batch_normalization(
                x, moving_mean, moving_var, beta, gamma, epsilon)

        # 兼容 Python bool 和张量两种情况
        if isinstance(is_training, bool):
            # 纯 Python 分支（你现在就是这种调用方式）
            return train_bn() if is_training else infer_bn()
        else:
            # 图中布尔张量，用 tf.cond 动态选择
            cond_pred = is_training
            if cond_pred.dtype is not tf.bool:
                cond_pred = tf.cast(cond_pred, tf.bool)
            return tf.cond(cond_pred, train_bn, infer_bn)


def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def conv2d_sn(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(x, filter=spectral_norm(W), strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1])

        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, W) + b


def init_embedding(size, dimension, stddev=0.01, scope="embedding"):
    with tf.variable_scope(scope):
        return tf.get_variable("E", [size, 1, 1, dimension], tf.float32,
                               tf.random_normal_initializer(stddev=stddev))


def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        batch_size, output_filters = shape[0], shape[-1]
        scale = tf.get_variable("scale", [labels_num, output_filters], tf.float32, tf.constant_initializer(1.0))
        shift = tf.get_variable("shift", [labels_num, output_filters], tf.float32, tf.constant_initializer(0.0))

        mu, sigma = tf.nn.moments(x, [1, 2], keep_dims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)

        batch_scale = tf.reshape(tf.nn.embedding_lookup([scale], ids=ids), [batch_size, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup([shift], ids=ids), [batch_size, 1, 1, output_filters])

        z = norm * batch_scale + batch_shift
        return z


def one_hot(indices, depth):
    result = [[0 for x in range(depth)] for y in range(len(indices))]
    for i in range(len(indices)):
        result[i][indices[i]] = 1

    return result


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):

        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


# spectrum normalization: https://github.com/taki0112/Spectral_Normalization-Tensorflow


def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def max_pooling(x) :
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')

# self attention gan: https://github.com/taki0112/Self-Attention-GAN-Tensorflow/
