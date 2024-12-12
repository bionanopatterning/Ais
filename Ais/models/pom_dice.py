import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, concatenate, Dropout
from tensorflow.keras.optimizers import Adam


title = "cryoPom-comb"
include = True


def dice_loss(y_true, y_pred, epsilon=1e-6):
    # Flatten the tensors to make it easier to compute the dice score
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Compute the Dice coefficient
    numerator = 2 * tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f + y_pred_f)

    dice_coeff = (numerator + epsilon) / (denominator + epsilon)
    dice_loss = 1 - dice_coeff
    return dice_loss


def create(input_shape, output_dimensionality=1):
    drop_rate = 0.15
    drop_rate_bottleneck = 0.3
    inputs = Input(input_shape)

    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(drop_rate)(pool3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(drop_rate)(pool4)

    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(drop_rate_bottleneck)(conv5)

    # Up Block 1
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    drop6 = Dropout(drop_rate)(conv6)


    # Up Block 2
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop6)
    merge7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    drop7 = Dropout(drop_rate)(conv7)

    # Up Block 3
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop7)
    merge8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    drop8 = Dropout(drop_rate)(conv8)

    # Up Block 4
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop8)
    merge9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output Layer
    output = Conv2D(output_dimensionality, (1, 1), activation='sigmoid')(conv9)

    # Create the model
    model = Model(inputs=[inputs], outputs=[output])

    # Compile the model with a suitable optimizer and loss function
    model.compile(optimizer=Adam(learning_rate=5e-5), loss=dice_loss)

    return model
