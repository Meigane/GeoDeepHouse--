import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    metrics = {
        'MAE': tf.keras.metrics.MeanAbsoluteError(),
        'MSE': tf.keras.metrics.MeanSquaredError(),
        'MAPE': tf.keras.metrics.MeanAbsolutePercentageError()
    }
    
    results = {}
    y_pred = model.predict(X_test)
    
    for name, metric in metrics.items():
        metric.update_state(y_test, y_pred)
        results[name] = metric.result().numpy()
    
    return results 