from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

title = "VGGNet L"
include = True

def masked_bxe(y_true, y_pred, border=16, ignore_label=2.0, epsilon=1e-6):
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

    return bce

def create(input_shape):
    inputs = Input(input_shape)

    # Block 1
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 2
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block 3
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Block 4
    conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool3)
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = BatchNormalization()(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    pool4 = Dropout(0.25)(pool4)

    # Upsampling and Decoding
    up1 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(pool4)
    up1 = Conv2D(1024, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)

    up2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(up1)
    up2 = Conv2D(512, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up2)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)

    up4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)

    output = Conv2D(1, (1, 1), activation='sigmoid')(up4)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss=masked_bxe)

    return model