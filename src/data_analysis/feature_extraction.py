import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import pyproj
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm
import json
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    # 设置文件路径
    base_path = Path(__file__).resolve().parent.parent.parent
    db_path = base_path / "data/processed/GisProject.db"
    poi_path = base_path / "data/gis_data/poi/上海市POI数据.csv"
    subway_path = base_path / "data/gis_data/subway.json"
    
    # 加载小区数据
    conn = sqlite3.connect(db_path)
    communities = pd.read_sql("SELECT * FROM communities", conn)
    
    # 加载POI数据
    poi_df = pd.read_csv(poi_path)
    
    # 加载地铁数据
    with open(subway_path, 'r', encoding='utf-8') as f:
        subway_data = json.load(f)
    
    subway_rows = []
    for feature in subway_data['l']:
        station = feature['st']
        for st in station:
            lon, lat = map(float, st['sl'].split(','))
            subway_rows.append({
                '名称': feature['ln'] + st['n'],
                '大类': '地铁站',
                '经度': lon,
                '纬度': lat
            })
    
    subway_df = pd.DataFrame(subway_rows)
    
    # 合并POI和地铁数据
    poi_df = pd.concat([
        poi_df,
        subway_df[['名称', '大类', '经度', '纬度']]
    ], ignore_index=True)
    
    return communities, poi_df, conn

def process_community_batch(args):
    """处理一批小区数据"""
    community_batch, poi_gdf, city_center_point = args
    features = []
    
    for _, community in community_batch.iterrows():
        community_point = Point(community['x'], community['y'])
        community_gdf = gpd.GeoDataFrame(
            {'geometry': [community_point]}, 
            crs='EPSG:4326'
        ).to_crs('EPSG:32649')
        
        # 计算到市中心的距离
        dist_to_center = community_gdf.distance(city_center_point)[0] / 1000
        
        # 计算POI密度
        poi_counts = {}
        for poi_type in ['地铁站', '科教文化', '医疗保健', '购物消费', '商务住宅', '生活服务']:
            type_pois = poi_gdf[poi_gdf['大类'] == poi_type]
            if not type_pois.empty:
                distances = type_pois.distance(community_gdf.geometry[0])
                poi_counts[poi_type] = sum(distances <= 2000)
                if poi_type == '地铁站':
                    nearest_subway_dist = distances.min()
            else:
                poi_counts[poi_type] = 0
                if poi_type == '地铁站':
                    nearest_subway_dist = np.nan
        
        features.append({
            '小区ID': community['id'],
            '距离市中心': dist_to_center,
            '距离最近地铁': nearest_subway_dist,
            '地铁站数量': poi_counts['地铁站'],
            '教育资源密度': poi_counts['科教文化'],
            '医疗设施密度': poi_counts['医疗保健'],
            '商业设施密度': sum([poi_counts[t] for t in ['购物消费', '商务住宅', '生活服务']])
        })
    
    return features

def optimize_calculation():
    """使用并行处理优化计算"""
    print("开始加载数据...")
    communities, poi_df, conn = load_data()
    
    # # 预先计算区域平均房价
    # print("计算区域平均房价...")
    # area_prices = communities.groupby('区域')['单价'].mean().to_dict()
    
    # 转换POI数据为GeoDataFrame
    print("转换数据格式...")
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=[Point(xy) for xy in zip(poi_df['经度'], poi_df['纬度'])],
        crs='EPSG:4326'
    ).to_crs('EPSG:32649')
    
    # 创建市中心点
    city_center = Point(121.4737, 31.2304)  # 人民广场
    city_center_gdf = gpd.GeoDataFrame(
        {'geometry': [city_center]}, 
        crs='EPSG:4326'
    ).to_crs('EPSG:32649')
    city_center_point = city_center_gdf.geometry[0]
    
    # 分批处理
    BATCH_SIZE = 200
    batches = [communities[i:i+BATCH_SIZE] for i in range(0, len(communities), BATCH_SIZE)]
    
    print("开始计算特征...")
    all_features = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_community_batch, 
                (batch, poi_gdf, city_center_point)
            ) 
            for batch in batches
        ]
        
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="处理小区批次"
        ):
            batch_features = future.result()
            all_features.extend(batch_features)
    
    # 转换为DataFrame
    features_df = pd.DataFrame(all_features)
    
    # # 添加区域平均房价
    # features_df['区域平均房价'] = communities['区域'].map(area_prices)
    
    # 将特征存入数据库
    print("保存特征到数据库...")
    features_df.to_sql('location_features', conn, if_exists='replace', index=False)
    
    # 更新communities表
    for column in features_df.columns:
        if column != '小区ID':
            sql = f"""
            ALTER TABLE communities ADD COLUMN IF NOT EXISTS {column} REAL;
            """
            conn.execute(sql)
            
            sql = f"""
            UPDATE communities 
            SET {column} = (
                SELECT {column} 
                FROM location_features 
                WHERE location_features.小区ID = communities.id
            );
            """
            conn.execute(sql)
    
    conn.commit()
    conn.close()
    
    return features_df

def main():
    """主函数"""
    try:
        features = optimize_calculation()
        print("\n特征统计信息:")
        print(features.describe())
        print("\n特征计算和保存完成!")
        
    except Exception as e:
        print(f"特征提取过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
