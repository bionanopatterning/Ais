import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv2DTranspose, Conv2D, BatchNormalization, concatenate, Dropout, Lambda
from tensorflow.keras.optimizers import Adam


title = "ezm-3d-17d"
include = True
dimensionality = 3

import tensorflow as tf

def dice_loss(y_true, y_pred, ignore_label=2.0, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.cast(tf.not_equal(y_true, ignore_label), tf.float32)
    y_true_clean = tf.where(tf.equal(y_true, ignore_label), 0.0, y_true)

    y_true_f = tf.reshape(y_true_clean * mask, [-1])
    y_pred_f = tf.reshape(y_pred * mask, [-1])
    mask_f = tf.reshape(mask, [-1])

    numerator = 2.0 * tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f + y_pred_f)
    valid = tf.reduce_sum(mask_f)

    dice_coeff = tf.where(tf.equal(valid, 0.0), 1.0, (numerator + epsilon) / (denominator + epsilon))
    return 1.0 - dice_coeff


def combined_loss(y_true, y_pred, border=16, ignore_label=2.0, epsilon=1e-6):
    if border > 0:
        y_true = y_true[:, border:-border, border:-border, ...]
        y_pred = y_pred[:, border:-border, border:-border, ...]

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.cast(tf.not_equal(y_true, ignore_label), tf.float32)
    y_true_clean = tf.where(tf.equal(y_true, ignore_label), 0.0, y_true)

    bce = tf.keras.losses.binary_crossentropy(y_true_clean, y_pred)  # shape [B,H,W]
    if tf.rank(mask) == 4: mask_bce = tf.squeeze(mask, axis=-1)
    else: mask_bce = mask

    bce = tf.reduce_sum(bce * mask_bce) / (tf.reduce_sum(mask_bce) + epsilon)

    dice = dice_loss(y_true, y_pred, ignore_label=ignore_label, epsilon=epsilon)

    return 0.3 * bce + 0.7 * dice


def create(input_shape, output_dimensionality=1):
    input_shape = (*input_shape, 1)
    print(f'create easymode-3d-center input_shape: {input_shape}')
    drop_rate_bottleneck = 0.25
    inputs = Input(input_shape)

    # 3D encoder
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(drop_rate_bottleneck)(conv5)

    # Extract center slice from all levels
    def slice_center_z(t):
        return t[:, :, :, t.shape[3] // 2, :]

    skip1 = Lambda(slice_center_z)(conv1)  # 128x128x64
    skip2 = Lambda(slice_center_z)(conv2)  # 64x64x128
    skip3 = Lambda(slice_center_z)(conv3)  # 32x32x256
    skip4 = Lambda(slice_center_z)(conv4)  # 16x16x512
    bottleneck = Lambda(slice_center_z)(drop5)  # 8x8x1024

    # 2D decoder with skip connections
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    merge6 = concatenate([up6, skip4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, skip3], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, skip2], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, skip1], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)

    output = Conv2D(output_dimensionality, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=5e-5), loss=combined_loss)

    return model