from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam

title = "UNet deep 2"


def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)

    # decoding
    up4 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)

    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
