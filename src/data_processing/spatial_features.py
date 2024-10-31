def extract_spatial_features(df, poi_data):
    features = {
        "距离市中心": calculate_distance_to_center,
        "地铁可达性": calculate_subway_accessibility,
        "教育设施密度": calculate_education_density,
        "商业设施密度": calculate_commercial_density,
        "医疗设施密度": calculate_medical_density
    }
    return features 