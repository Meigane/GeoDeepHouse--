import tensorflow as tf

def create_dcn_model(input_dim):
    # Deep & Cross Network
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    
    # Cross Network
    cross_layer = CrossLayer()(input_layer)
    
    # Deep Network
    deep = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    
    # Combine
    combined = tf.keras.layers.concatenate([cross_layer, deep])
    output = tf.keras.layers.Dense(1)(combined)
    
    return tf.keras.Model(inputs=input_layer, outputs=output) 