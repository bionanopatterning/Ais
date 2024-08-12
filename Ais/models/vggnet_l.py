from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

title = "VGGNet L"
include = True

def create(input_shape):
    inputs = Input(input_shape)

    # Block 1
    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool1 = Dropout(0.25)(pool1)

    # Block 2
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool2 = Dropout(0.25)(pool2)

    # Block 3
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
    pool3 = Dropout(0.25)(pool3)

    # Block 4
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = BatchNormalization()(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    pool4 = Dropout(0.25)(pool4)

    # Upsampling and Decoding
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(pool4)
    up1 = Conv2D(512, (3, 3), activation='relu', padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(0.25)(up1)

    up2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(up1)
    up2 = Conv2D(512, (3, 3), activation='relu', padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(0.25)(up2)

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up2)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(0.25)(up3)

    up4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Dropout(0.25)(up4)

    output = Conv2D(1, (1, 1), activation='sigmoid')(up4)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model