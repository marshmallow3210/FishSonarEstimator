import tensorflow as tf
import numpy as np
import cv2
import os

class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=4, axis=-1, epsilon=1e-5, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[self.axis],),
                                     initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                    initializer='zeros', trainable=True, name='beta')
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        G = self.groups
        inputs = tf.reshape(inputs, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(inputs, [1, 2, 4], keepdims=True)
        inputs = (inputs - mean) / tf.sqrt(var + self.epsilon)
        inputs = tf.reshape(inputs, [N, H, W, C])
        return self.gamma * inputs + self.beta


def se_block(x, ratio=8):
    ch = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(ch // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(ch, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, ch))(se)
    return tf.keras.layers.Multiply()([x, se])

def mha_counter(height, width, channel, network_depth, conv_layer, head_nums, d_model, pe_type='None', pe_channel=8):
    inputs = tf.keras.layers.Input(shape=(height, width, 1))
    x = inputs

    for i in range(network_depth):
        b1 = tf.keras.layers.Conv2D(channel, 3, padding='same', activation='relu')(x)
        b2 = tf.keras.layers.Conv2D(channel, 3, padding='same', dilation_rate=2, activation='relu')(x)
        b3 = tf.keras.layers.Conv2D(channel, 3, padding='same', dilation_rate=3, activation='relu')(x)
        merged = tf.keras.layers.Add()([b1, b2, b3])

        x = GroupNormalization(groups=4)(merged)
        x = tf.keras.layers.Activation('relu')(x)
        x = se_block(x, ratio=8)
        x = tf.keras.layers.MaxPooling2D()(x)
        channel += 4

    A = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    Q = tf.keras.layers.Conv2D(d_model // head_nums, 3, padding="same", activation='relu')(A)
    K = tf.keras.layers.Conv2D(d_model // head_nums, 3, padding="same", activation='relu')(A)
    V = tf.keras.layers.Conv2D(d_model // head_nums, 3, padding="same", activation='relu')(A)

    Q_reshaped = tf.keras.layers.Reshape(target_shape=(-1, Q.shape[-1]))(Q)
    K_reshaped = tf.keras.layers.Reshape(target_shape=(-1, K.shape[-1]))(K)
    V_reshaped = tf.keras.layers.Reshape(target_shape=(-1, V.shape[-1]))(V)

    Q_reshaped = tf.keras.layers.LayerNormalization()(Q_reshaped)
    K_reshaped = tf.keras.layers.LayerNormalization()(K_reshaped)
    V_reshaped = tf.keras.layers.LayerNormalization()(V_reshaped)

    mha = tf.keras.layers.MultiHeadAttention(num_heads=head_nums, key_dim=d_model // head_nums)
    O = mha(Q_reshaped, V_reshaped, K_reshaped)
    E = tf.keras.layers.Reshape(target_shape=V.shape[1:])(O)

    x = tf.keras.layers.Concatenate()([x, tf.keras.layers.UpSampling2D(size=(2, 2))(E)])
    x0 = x
    x1 = tf.keras.layers.Conv2D(channel, 3, activation='relu', padding='same')(x0)
    x2 = tf.keras.layers.Conv2D(channel, 3, activation='relu', padding='same')(x1)
    x3 = tf.keras.layers.Conv2D(channel, 3, activation='relu', padding='same')(x2)
    x = tf.keras.layers.Concatenate()([x0, x1, x2, x3])
    x = tf.keras.layers.Conv2D(channel, 1, activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation='relu', padding='same', name='density_map')(x)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs], name='mha_counter')

MODEL_PATH = "kernel_9_counter_weights.h5"
model = mha_counter(height=576, width=320, channel=24, network_depth=3, conv_layer=5, head_nums=2, d_model=16)
model.load_weights(MODEL_PATH)
print("模型架構與權重載入成功！")

def preprocess_image(image_path, target_size=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"無法讀取影像檔案：{image_path}")

    image = image.astype(np.float32) / 255.0
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=(0, -1))  # (1, H, W, 1)
    return image

def estimate_fish(image_array):
    prediction = model.predict(image_array)
    estimated_count = np.sum(prediction)
    return float(estimated_count)

def estimate_from_file(image_path, target_size=(320, 576)):
    image_array = preprocess_image(image_path, target_size)
    return estimate_fish(image_array)

if __name__ == "__main__":
    image_path = "201.png"  # 測試用影像
    try:
        estimated_count = estimate_from_file(image_path)
        print(f"預估魚隻數量: {estimated_count:.2f}")
    except Exception as e:
        print(f"發生錯誤: {e}")