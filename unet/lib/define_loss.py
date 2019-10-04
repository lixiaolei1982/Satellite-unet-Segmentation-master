from keras import backend as K
import tensorflow as tf


def balanced_cross_entropy(alpha=.60):  #。75

    def balanced_cross_entropy_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        #return -K.sum(alpha * K.pow(1.- pt_1,gamma) * K.log(K.epsilon() + pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

        return -K.sum(alpha * K.log( K.epsilon()+pt_1)) - K.sum(
            (1 - alpha) * K.log(1. - pt_0 +K.epsilon()))
    return balanced_cross_entropy_fixed


def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        '''
        只有tf.where(input, name=None)一种用法，在实际应用中发现了另外一种使用方法tf.where(input, a,
         b)，其中a，b均为尺寸一致的tensor，作用是将a中对应input中true的位置
        的元素值不变，其余元素进行替换，替换成b中对应位置的元素值
        '''
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        #return -K.sum(alpha * K.pow(1.- pt_1,gamma) * K.log(K.epsilon() + pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log( K.epsilon()+pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 +K.epsilon()))
    return focal_loss_fixed


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.sum(focal_loss)

    return binary_focal_loss_fixed


#'''
#smooth 参数防止分母为0
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)
