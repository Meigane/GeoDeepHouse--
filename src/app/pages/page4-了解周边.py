import streamlit as st  
import pandas as pd  
import geopandas as gpd  
from shapely.geometry import Point, Polygon  
import folium  
import sqlite3
from streamlit_folium import st_folium
from pathlib import Path
import json
import pyproj
from folium import Marker, FeatureGroup   

file_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" )
gis_path = file_path + "//gis_data"

def load_subway():
    f = gis_path + "//subway.json"  
    with open(f, 'r', encoding='utf-8') as f:  
        data = json.load(f)  
  
    rows = []  
    for feature in data['l']:  
        station = feature['st']  
        for st in station: 
            lon, lat = map(float, st['sl'].split(','))  
            geometry = Point(lon, lat)  
            properties = {"name": feature['ln'] + st['n']}
            rows.append({'geometry': geometry, **properties})  
  
    # 创建GeoDataFrame并设置正确的初始坐标系统
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')  
    # 转换到投影坐标系统（北京54坐标系）
    gdf = gdf.to_crs('EPSG:2435')
    return gdf  

@st.cache_data
def load_poi_data():  
    poi_files = [   
        "教育资源.shp",
        "医疗保健.shp"  
    ]  
      
    all_poi_data = None  
    common_columns = ['geometry','name']  
      
    for file_name in poi_files:  
        poi_data = gpd.read_file(gis_path + "//poi//" + file_name)
        # 确保POI数据使用正确的坐标系统
        if poi_data.crs is None:
            poi_data.set_crs('EPSG:4326', inplace=True)
        # 转换到投影坐标系统
        poi_data = poi_data.to_crs('EPSG:2435')
        
        poi_type = file_name.split('.')[0]  
        poi_data['type'] = poi_type  
        poi_data = poi_data[common_columns + ['type']]  
          
        if all_poi_data is None:  
            all_poi_data = poi_data  
        else:  
            all_poi_data = pd.concat([all_poi_data, poi_data], ignore_index=True)  
      
    return all_poi_data

@st.cache_data
def load_community_data():
    communities = gpd.read_file(gis_path+"//communities.shp")
    # 确保小区数据使用正确的坐标系统
    if communities.crs is None:
        communities.set_crs('EPSG:4326', inplace=True)
    # 转换到投影坐标系统
    communities = communities.to_crs('EPSG:2435')
    return communities

def find_nearest_pois(community_point, type, poi_data):  
    nearest_pois = {}
    nearest_poi_within = {}  
    i = 1 
    
    # 创建大地测量计算器
    geod = pyproj.Geod(ellps='WGS84')
    
    # 将community_point转换为GeoSeries以保持坐标系统信息
    community_geoseries = gpd.GeoSeries([community_point.geometry], crs='EPSG:2435')
    community_point_wgs84 = community_geoseries.to_crs('EPSG:4326')[0]
    
    # 获取社区点的经纬度
    community_lon = community_point_wgs84.x
    community_lat = community_point_wgs84.y
    
    # 路网修正系数（考虑实际道路距离比直线距离长）
    ROAD_FACTOR = 1.7  # 一般城市路网修正系数在1.3-1.5之间
    
    for poi_type in type:
        if i > 3:
            break  
        type_pois = poi_data[poi_data['type'] == poi_type].copy()
        
        # 批量计算距离以提高效率
        # 将所有POI点转换为WGS84
        type_pois_wgs84 = type_pois.to_crs('EPSG:4326')
        
        # 向量化计算距离
        lons = type_pois_wgs84.geometry.x
        lats = type_pois_wgs84.geometry.y
        
        # 使用numpy广播计算距离
        distances = []
        # 每次处理一批POI点以平衡内存使用和性能
        BATCH_SIZE = 1000
        for start_idx in range(0, len(type_pois), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(type_pois))
            batch_lons = lons[start_idx:end_idx]
            batch_lats = lats[start_idx:end_idx]
            
            # 批量计算大地测量距离
            _, _, batch_distances = geod.inv(
                [community_lon] * len(batch_lons),
                [community_lat] * len(batch_lons),
                batch_lons,
                batch_lats
            )
            distances.extend([abs(d) * ROAD_FACTOR for d in batch_distances])
        
        type_pois['distance'] = distances
        
        # 考虑路网修正后的2000米实际上相当于直线距离约1430米
        adjusted_radius = 2000 / ROAD_FACTOR 
        all_poi_within = type_pois[type_pois['distance'] <= 3000]
        type_pois_within_2km = all_poi_within[type_pois['distance'] <= adjusted_radius]
        
        # 将距离四舍五入到整数
        type_pois_within_2km['distance'] = type_pois_within_2km['distance'].round(0) * ROAD_FACTOR
        all_poi_within['distance'] = all_poi_within['distance'].round(0)
        
        # 按距离排序
        type_pois_within_2km = type_pois_within_2km.sort_values('distance')
        all_poi_within = all_poi_within.sort_values('distance')
        
        nearest_pois[poi_type] = type_pois_within_2km
        nearest_poi_within[poi_type] = all_poi_within
        i += 1
      
    if '交通' in type and '交通' in nearest_poi_within:  
        nearest_subway = nearest_poi_within['交通'].iloc[0] if not nearest_poi_within['交通'].empty else None
        if nearest_subway is not None:
            nearest_subway_info = nearest_subway.to_dict()
        else:
            nearest_subway_info = {'name': '无地铁信息', 'distance': None}
    else:  
        nearest_subway_info = {'name': '无地铁信息', 'distance': None}  

    return nearest_pois, nearest_subway_info 

def load_all_pois():  
    # 加载POI数据（不包括地铁）  
    poi_data = load_poi_data()    
      
    # 加载地铁数据  
    subway_data = load_subway()  
    # 给地铁数据添加一个'type'列，标记为'交通'  
    subway_data['type'] = '交通'  
      
    # 将地铁数据添加到POI数据中  
    all_pois = pd.concat([poi_data, subway_data], ignore_index=True)  
      
    # 转换为GeoDataFrame  
    all_pois = gpd.GeoDataFrame(all_pois, geometry='geometry', crs='EPSG:2435')  
      
    return all_pois    

def run(): 
    st.title("小区周边查询") 
    communities = load_community_data()
    # 使用load_all_pois替代load_poi_data
    all_pois = load_all_pois()
    filtered_communities = communities
    selected_row = None
    with st.form("my_form1"):  
        # 这里假设 filtered_communities 已经在某处被赋值且不为空  
        if filtered_communities is not None and not filtered_communities.empty:  
            selected_community = st.selectbox("选择进一步想要查看的小区", filtered_communities['name'].sort_values())  
            options = ["教育资源", "医疗保健", "交通"]  
            selected_options = st.multiselect('请选择你感兴趣的选项', options)  
            submitted2 = st.form_submit_button("提交")  
  
            nearest_pois = None  
            nearest_subway_info = None  
  
            if submitted2:  
                selected_row = filtered_communities[filtered_communities['name'] == selected_community].iloc[0]   
                nearest_pois, nearest_subway_info = find_nearest_pois(selected_row, selected_options, all_pois)
  
    if nearest_pois is not None:  
        st.subheader("两公里内的POI信息：")  
        for poi_type, pois in nearest_pois.items():  
            if poi_type in selected_options:  
                st.write(f"{poi_type}：")  
                #st.write(pois.to_string(index=False))  
                st.dataframe(pois[['name', 'distance']])
  
        st.subheader("最近的地铁信息：")  
        st.write(f"名称：{nearest_subway_info['name']}，距离：{nearest_subway_info['distance']}米")
    # if filtered_communities is not None and not filtered_communities.empty:
    #     m = folium.Map(location=[selected_row.x, selected_row.y], zoom_start=13)  # 示例中心点和缩放级别  
        
    #     # 创建一个FeatureGroup来存储POI  
    #     poi_layer = FeatureGroup(name='poi图层', control=True)
    #     folium.TileLayer(tiles = 'Gaode.Normal',name="高德地图",control=True).add_to(m)  
        
    #     # 遍历DataFrame中的POI，并将它们添加到FeatureGroup中
    #     for pois in nearest_pois.values():
    #         for i, poi in enumerate(pois):
    #             print(poi)  
    #             x = float(poi[0].x)  
    #             y = float(poi[0].y)  
    #             name = nearest_pois['name'][i]  
    #             ty = nearest_pois['type'][i]  
    #             d = nearest_pois['distance'][i]  
            
    #             html = f"""  
    #                 <h4>{name}</h4>  
    #                 <h6>类型：{ty}<br>距离{d}</h6>  
    #             """  
    #             popup = folium.Popup(html, max_width=265)  
            
    #             marker = Marker(location=[y, x], popup=popup)  
    #             marker.add_to(poi_layer)  
        
    #     # 将FeatureGroup添加到地图上  
    #     poi_layer.add_to(m)
    #     st_folium(m, width=700, height=500)    
  
if __name__ == "__main__":  
    run()