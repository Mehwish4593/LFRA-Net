"""LFRA-Net: a lightweight retinal vessel segmentation network (174,097 parameters).

Transcribed from LFRA-Net.ipynb cells 4-6 without alteration, so the released
.h5 weights load by name. Built under tf_keras (legacy Keras) because the
weights are Keras 2.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model, backend as K
from tf_keras.layers import (Conv2D, BatchNormalization, Activation, Multiply,
                             Reshape, Lambda, GlobalMaxPool2D, GlobalAveragePooling2D)


def focal_modulation_block(inputs, gamma=2.0, alpha=0.25):
    num_channels = inputs.shape[-1]
    mean = layers.GlobalAveragePooling2D()(inputs)
    max_val = layers.GlobalMaxPooling2D()(inputs)
    modulation = (max_val - mean) * alpha
    modulation = layers.Reshape((1, 1, num_channels))(modulation)
    modulation = layers.Conv2D(num_channels, 1, activation="sigmoid")(modulation)
    scaled = layers.Multiply()([inputs, modulation])
    return layers.Lambda(lambda x: x ** gamma)(scaled)


def focal_modulation_context_aggregation_block(inputs, filters):
    filters = int(filters)
    conv1 = layers.Conv2D(filters, 1, padding="same", activation="relu")(inputs)
    conv2 = layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    conv3 = layers.Conv2D(filters, 5, padding="same", activation="relu")(inputs)
    gate1 = layers.Conv2D(filters, 1, activation="sigmoid")(inputs)
    gate2 = layers.Conv2D(filters, 1, activation="sigmoid")(inputs)
    gate3 = layers.Conv2D(filters, 1, activation="sigmoid")(inputs)
    agg = layers.Add()([layers.Multiply()([conv1, gate1]),
                        layers.Multiply()([conv2, gate2]),
                        layers.Multiply()([conv3, gate3])])
    fm = focal_modulation_block(agg)
    query = layers.Conv2D(filters, 1, activation="relu")(inputs)
    modulator = layers.Conv2D(filters, 1, activation="relu")(fm)
    return layers.Add()([inputs, layers.Multiply()([query, modulator])])


def ana_attention_block(inputs, n_classes, k):
    shape = K.int_shape(inputs)
    F = Conv2D(k * n_classes, 3, padding="same")(inputs)
    F = BatchNormalization()(F)
    F1 = Activation("relu")(F)
    x1 = GlobalMaxPool2D()(F1)
    x2 = GlobalAveragePooling2D()(F1)
    x = Multiply()([x1, x2])
    x = Reshape((n_classes, k))(x)
    S = Lambda(lambda x: K.mean(x, axis=-1, keepdims=False))(x)
    x = Reshape((shape[1], shape[2], n_classes, k))(F1)
    x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=False))(x)
    x = Multiply()([S, x])
    M = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x)
    return Multiply()([inputs, M])


def conv_block(inp, out_channels, dropout_rate=0):
    a = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(inp)
    b = layers.Conv2D(out_channels, 3, padding="same", use_bias=False)(inp)
    c = layers.Conv2D(out_channels, 3, padding="same", use_bias=False, dilation_rate=2)(inp)
    x = a + b + c
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(out_channels, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    return layers.LeakyReLU(alpha=0.1)(x)


def up_block(inp, out_channels):
    x = layers.Conv2DTranspose(out_channels, 2, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(alpha=0.1)(x)


def down_block(inp, out_channels):
    x = layers.Conv2D(out_channels, 2, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(alpha=0.1)(x)


def build_FR_UNet(input_shape=(512, 512, 3), num_classes=1, feature_scale=2, dropout=0.2):
    inputs = keras.Input(shape=input_shape)
    f = [int(x / feature_scale) for x in [16, 32, 64]]

    c1 = conv_block(inputs, f[0], dropout); p1 = down_block(c1, f[0])
    c2 = conv_block(p1, f[1], dropout);     p2 = down_block(c2, f[1])
    c3 = conv_block(p2, f[2], dropout);     p3 = down_block(c3, f[2])

    x1_up = up_block(c3, f[2])
    x2_s = ana_attention_block(c2, 1, 16)
    x2 = layers.concatenate([x2_s, x1_up])
    x2_up = up_block(c2, f[1])
    x3_s = ana_attention_block(c1, 1, 16)
    x3 = layers.concatenate([x3_s, x2_up])
    x3_up = up_block(x3, f[0])

    fm = focal_modulation_context_aggregation_block(p3, 32)
    c4 = conv_block(fm, f[2], dropout)
    u6 = up_block(c4, f[2]); c6 = conv_block(u6, f[2], dropout)
    u7 = up_block(c6, f[1]); u7 = layers.concatenate([u7, x2]); c7 = conv_block(u7, f[1], dropout)
    u8 = up_block(c7, f[0]); u8 = layers.concatenate([u8, x3]); c8 = conv_block(u8, f[0], dropout)

    out = layers.Conv2D(num_classes, 1, activation="sigmoid")(c8)
    return Model(inputs=[inputs], outputs=[out])
