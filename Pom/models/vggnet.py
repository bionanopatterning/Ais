from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

title = "VGGNet"
include = True

def create(input_shape):
    inputs = Input(input_shape)

    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 2
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block 3
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Upsampling and Decoding
    up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(pool3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

    up3 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(conv8)
    output = Conv2D(1, (1, 1), activation='sigmoid')(up3)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model