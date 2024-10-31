import tensorflow as tf

def train_model(X_train, y_train, config):
    # 1. 创建模型
    model = create_dnn_model(X_train.shape[1])
    
    # 2. 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    # 3. 训练模型
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks()
    )
    
    return model, history 