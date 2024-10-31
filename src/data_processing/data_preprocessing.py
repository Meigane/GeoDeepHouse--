def preprocess_data(df):
    # 1. 处理缺失值
    df = handle_missing_values(df)
    
    # 2. 特征工程
    df = add_location_features(df)
    df = add_time_features(df)
    df = add_structural_features(df)
    
    # 3. 特征编码
    df = encode_categorical_features(df)
    
    # 4. 特征缩放
    df = scale_features(df)
    
    return df 