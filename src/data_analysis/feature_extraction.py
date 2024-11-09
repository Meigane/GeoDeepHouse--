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
from tqdm.auto import tqdm
import sys
import time

def load_data():
    """加载数据并进行预处理以优化计算速度"""
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
    
    # POI数据预过滤 - 只保留上海市数据
    poi_df = poi_df[poi_df['城市'] == '上海市']
    
    # 预先过滤掉明显超出范围的POI
    # 获取上海市的大致边界
    shanghai_bounds = {
        'lon_min': 120.8,
        'lon_max': 122.0,
        'lat_min': 30.7,
        'lat_max': 31.9
    }
    
    poi_df = poi_df[
        (poi_df['经度'] >= shanghai_bounds['lon_min']) &
        (poi_df['经度'] <= shanghai_bounds['lon_max']) &
        (poi_df['纬度'] >= shanghai_bounds['lat_min']) &
        (poi_df['纬度'] <= shanghai_bounds['lat_max'])
    ]
    
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
    """优化的批处理函数"""
    community_batch, poi_gdf, city_center_point = args
    features = []
    
    # 验证输入数据
    if not isinstance(poi_gdf, gpd.GeoDataFrame):
        raise TypeError("poi_gdf must be a GeoDataFrame")
    if not isinstance(city_center_point, Point):
        raise TypeError("city_center_point must be a Point")
    
    # 创建进度条
    progress_bar = tqdm(
        total=len(community_batch),
        desc=f"批次处理进度(size={len(community_batch)})",
        position=1,
        leave=False
    )
    
    # 为每个小区创建2km缓冲区
    buffer_distance = 2000  # 2公里
    
    # 确保坐标数据有效
    valid_communities = community_batch[
        community_batch['x'].notna() & 
        community_batch['y'].notna() &
        (community_batch['x'] != 0) & 
        (community_batch['y'] != 0)
    ]
    
    # 创建有效的Point对象
    community_points = []
    for _, row in valid_communities.iterrows():
        try:
            point = Point(float(row['x']), float(row['y']))
            if point.is_valid:
                community_points.append(point)
        except (ValueError, TypeError) as e:
            print(f"无效的坐标: x={row['x']}, y={row['y']}, error={str(e)}")
            continue
    
    if not community_points:
        print("没有有效的小区点")
        return []
    
    # 创建GeoDataFrame
    community_gdf = gpd.GeoDataFrame(
        geometry=community_points,
        crs='EPSG:4326'
    ).to_crs('EPSG:32649')
    
    # 创建缓冲区
    try:
        buffers = community_gdf.buffer(buffer_distance)
        buffer_union = buffers.unary_union
        
        # 使用空间索引加速查询
        if hasattr(poi_gdf, 'sindex'):
            possible_matches_index = list(poi_gdf.sindex.intersection(buffer_union.bounds))
            poi_subset = poi_gdf.iloc[possible_matches_index].copy()
        else:
            poi_subset = poi_gdf.copy()
            
    except Exception as e:
        print(f"创建缓冲区时出错: {str(e)}")
        return []
    
    for idx, community in valid_communities.iterrows():
        try:
            # 创建单个小区的GeoDataFrame
            community_point = Point(float(community['x']), float(community['y']))
            if not community_point.is_valid:
                continue
                
            community_gdf_single = gpd.GeoDataFrame(
                {'geometry': [community_point]}, 
                crs='EPSG:4326'
            ).to_crs('EPSG:32649')
            
            # 创建缓冲区
            buffer = community_gdf_single.buffer(buffer_distance).iloc[0]
            
            # 使用空间查询找到附近的POI
            nearby_pois = poi_subset[poi_subset.intersects(buffer)]
            
            # 计算到市中心的距离
            dist_to_center = float(community_gdf_single.distance(city_center_point)[0]) / 1000
            
            # 计算POI密度
            poi_counts = {}
            for poi_type in ['地铁站', '科教文化', '医疗保健', '购物消费', '商务住宅', '生活服务']:
                type_pois = nearby_pois[nearby_pois['大类'] == poi_type]
                if not type_pois.empty:
                    distances = type_pois.distance(community_gdf_single.geometry[0])
                    poi_counts[poi_type] = int(sum(distances <= 2000))
                    if poi_type == '地铁站':
                        nearest_subway_dist = float(distances.min())
                else:
                    poi_counts[poi_type] = 0
                    if poi_type == '地铁站':
                        nearest_subway_dist = np.nan
            
            features.append({
                '小区': community['name'],
                '距离市中心': dist_to_center,
                '距离最近地铁': nearest_subway_dist,
                '地铁站数量': poi_counts['地铁站'],
                '教育资源密度': poi_counts['科教文化'],
                '医疗设施密度': poi_counts['医疗保健'],
                '商业设施密度': sum([poi_counts[t] for t in ['购物消费', '商务住宅', '生活服���']])
            })
            
            progress_bar.update(1)
            progress_bar.set_postfix({'当前小区': str(community['name'])[:10]})
            
        except Exception as e:
            print(f"处理小区 {community['name']} 时出错: {str(e)}")
            continue
    
    progress_bar.close()
    return features

def calculate_features_gpu(communities, poi_gdf, city_center_point):
    """使用GPU计算特征（如果可用）"""
    try:
        import cudf
        import cuspatial
        print("使用GPU加速计算...")
        
        # 转换数据到GPU
        communities_gpu = cudf.from_pandas(communities)
        poi_gpu = cudf.from_pandas(poi_gdf)
        
        # GPU计算逻辑
        # ... (GPU specific implementation)
        
        return features_gpu.to_pandas()
    except ImportError:
        print("GPU库不可用，使用CPU计算...")
        return None

def optimize_calculation():
    """优化的主计算函数"""
    print("开始加载数据...")
    communities, poi_df, conn = load_data()
    
    print(f"总计需要处理 {len(communities)} 个小区")
    print(f"POI数据点数量: {len(poi_df)}")
    
    # 转换POI数据为GeoDataFrame
    print("转换数据格式...")
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=[Point(float(x), float(y)) for x, y in zip(poi_df['经度'], poi_df['纬度']) 
                 if pd.notna(x) and pd.notna(y)],
        crs='EPSG:4326'
    ).to_crs('EPSG:32649')
    
    # 创建市中心点
    city_center = Point(121.4737, 31.2304)
    city_center_gdf = gpd.GeoDataFrame(
        {'geometry': [city_center]}, 
        crs='EPSG:4326'
    ).to_crs('EPSG:32649')
    city_center_point = city_center_gdf.geometry[0]
    
    # 尝试使用GPU计算
    features_df = calculate_features_gpu(communities, poi_gdf, city_center_point)
    
    if features_df is None:
        # 如果GPU不可用，使用CPU计算
        print("使用CPU并行计算...")
        BATCH_SIZE = 100
        batches = [communities[i:i+BATCH_SIZE] for i in range(0, len(communities), BATCH_SIZE)]
        print(f"将分 {len(batches)} 批处理，每批 {BATCH_SIZE} 个小区")
        
        all_features = []
        main_progress = tqdm(total=len(batches), desc="总体进度", position=0)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    process_community_batch, 
                    (batch, poi_gdf, city_center_point)
                ) 
                for batch in batches
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_features = future.result()
                    all_features.extend(batch_features)
                    main_progress.update(1)
                except Exception as e:
                    print(f"批次处理失败: {str(e)}")
                    continue
        
        main_progress.close()
        features_df = pd.DataFrame(all_features)
    
    # 保存结果到数据库
    print("保存特征到数据库...")
    if not features_df.empty:
        # 先保存到临时表
        features_df.to_sql('location_features', conn, if_exists='replace', index=False)
        
        # 获取现有表结构
        existing_columns = pd.read_sql("PRAGMA table_info(communities);", conn)['name'].tolist()
        
        for column in features_df.columns:
            if column != '小区':
                try:
                    # 如果列不存在，添加列
                    if column not in existing_columns:
                        conn.execute(f"ALTER TABLE communities ADD COLUMN {column} REAL;")
                    
                    # 更新数据
                    conn.execute(f"""
                        UPDATE communities 
                        SET {column} = (
                            SELECT {column} 
                            FROM location_features 
                            WHERE location_features.小区 = communities.name
                        );
                    """)
                except Exception as e:
                    print(f"更新列 {column} 时出错: {str(e)}")
                    continue
        
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

def update_nearest_subway_distance():
    """更新距离最近地铁站为空的小区的地铁站距离"""
    try:
        # 设置文件路径
        base_path = Path(__file__).resolve().parent.parent.parent
        db_path = base_path / "data/processed/GisProject.db"
        subway_path = base_path / "data/gis_data/subway.json"
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        
        # 获取距离最近地铁为空的小区
        null_subway_communities = pd.read_sql("""
            SELECT name, x, y 
            FROM communities 
            WHERE 距离最近地铁 IS NULL OR 距离最近地铁 = ''
        """, conn)
        
        print(f"发现 {len(null_subway_communities)} 个需要更新地铁站距离的小区")
        
        # 加载地铁站数据
        with open(subway_path, 'r', encoding='utf-8') as f:
            subway_data = json.load(f)
        
        # 处理地铁站数据
        subway_points = []
        for feature in subway_data['l']:
            for station in feature['st']:
                lon, lat = map(float, station['sl'].split(','))
                subway_points.append({
                    'name': feature['ln'] + station['n'],
                    'geometry': Point(lon, lat)
                })
        
        # 创建地铁站GeoDataFrame
        subway_gdf = gpd.GeoDataFrame(
            subway_points,
            crs='EPSG:4326'
        ).to_crs('EPSG:32649')  # 转换到投影坐标系统
        
        # 创建进度条
        progress_bar = tqdm(total=len(null_subway_communities), desc="更新地铁站距离")
        
        # 创建大地测量计算器
        geod = pyproj.Geod(ellps='WGS84')
        
        # 批量处理小区
        batch_size = 100
        updates = []
        
        for i in range(0, len(null_subway_communities), batch_size):
            batch = null_subway_communities.iloc[i:i+batch_size]
            
            for _, community in batch.iterrows():
                try:
                    # 创建小区点
                    community_point = Point(community['x'], community['y'])
                    community_gdf = gpd.GeoDataFrame(
                        {'geometry': [community_point]}, 
                        crs='EPSG:4326'
                    ).to_crs('EPSG:32649')
                    
                    # 计算到所有地铁站的距离
                    distances = subway_gdf.geometry.distance(community_gdf.geometry[0])
                    min_distance = distances.min()
                    
                    # 添加到更新列表
                    updates.append({
                        'name': community['name'],
                        'distance': float(min_distance)
                    })
                    
                except Exception as e:
                    print(f"\n处理小区 {community['name']} 时出错: {str(e)}")
                    continue
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        print("\n开始更新数据库...")
        # 批量更新数据库
        for update in tqdm(updates, desc="保存到数据库"):
            try:
                conn.execute("""
                    UPDATE communities 
                    SET 距离最近地铁 = ? 
                    WHERE name = ?
                """, (update['distance'], update['name']))
            except Exception as e:
                print(f"\n更新小区 {update['name']} 时出错: {str(e)}")
                continue
        
        conn.commit()
        print(f"\n成功更新了 {len(updates)} 个小区的地铁站距离")
        
        # 显示更新后的统计信息
        stats = pd.read_sql("""
            SELECT 
                COUNT(*) as total_count,
                COUNT(距离最近地铁) as valid_count,
                AVG(距离最近地铁) as avg_distance,
                MIN(距离最近地铁) as min_distance,
                MAX(距离最近地铁) as max_distance
            FROM communities
        """, conn)
        
        print("\n更新后的统计信息：")
        print(stats)
        
        conn.close()
        
    except Exception as e:
        print(f"更新过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

if __name__ == "__main__":
    main()
