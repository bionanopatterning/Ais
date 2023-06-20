from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam

title = "UNet lite"

def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)

    # decoding
    up3 = Concatenate()([UpSampling2D(size=(2, 2))(conv2), conv1])
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    conv4 = Conv2D(1, (1, 1), activation='sigmoid')(conv3)

    # create the model
    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
