from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout, concatenate
from tensorflow.keras.optimizers import Adam

title = "UNet L"
include = True

def create(input_shape):
    inputs = Input(input_shape)

    # --- Encoder / Downsampling Path ---
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

    # --- Bridge / Bottleneck ---
    conv9 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Dropout(0.25)(conv10)

    # --- Decoder / Upsampling Path ---
    # Up Block 1 (corresponds to Block 4)
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv10)
    # Concatenate skip connection
    up1 = concatenate([up1, conv8], axis=-1)
    conv11 = Conv2D(512, (3, 3), activation='relu', padding='same')(up1)
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Dropout(0.25)(conv12)

    # Up Block 2 (corresponds to Block 3)
    up2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv12)
    up2 = concatenate([up2, conv6], axis=-1)
    conv13 = Conv2D(512, (3, 3), activation='relu', padding='same')(up2)
    conv13 = BatchNormalization()(conv13)
    conv14 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Dropout(0.25)(conv14)

    # Up Block 3 (corresponds to Block 2)
    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv14)
    up3 = concatenate([up3, conv4], axis=-1)
    conv15 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    conv15 = BatchNormalization()(conv15)
    conv16 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Dropout(0.25)(conv16)

    # Up Block 4 (corresponds to Block 1)
    up4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv16)
    up4 = concatenate([up4, conv2], axis=-1)
    conv17 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    conv17 = BatchNormalization()(conv17)
    conv18 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Dropout(0.25)(conv18)

    # --- Output Layer ---
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv18)

    # --- Build and Compile the Model ---
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model
