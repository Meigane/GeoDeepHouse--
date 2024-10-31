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
    # 读取GeoJSON文件  
    with open(f, 'r', encoding='utf-8') as f:  
        data = json.load(f)  
  
    # 初始化一个空列表来存储GeoDataFrame的行  
    rows = []  
  
    # 遍历特征  
    for feature in data['l']:
          
        # 从properties中提取信息  
        station = feature['st']  
          
        # 从sl字段解析经纬度并创建点几何对象 
        for st in station: 
            lon, lat = map(float, st['sl'].split(','))  
            geometry = Point(lon, lat)  

            properties = {"name":feature['ln'] + st['n']}
            # 将属性和几何对象添加到行列表中  
            rows.append({'geometry': geometry, **properties})  
  
    # 将行列表转换为GeoDataFrame  
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')  
    gdf = gdf.to_crs('EPSG:32649')
  
    return gdf  

@st.cache_data
def load_poi_data():  
    poi_files = [   
        "教育资源.shp" 
       ,"医疗保健.shp"  
    ]  
      
    all_poi_data = None  
    common_columns = ['geometry','name']  # 保留的通用列，至少包含geometry  
      
    for file_name in poi_files:  
        poi_data = gpd.read_file(gis_path + "//poi//" + file_name)  
        poi_type = file_name.split('.')[0]  
          
        # 确保每个POI数据都有一个'type'列来表示其类别  
        poi_data['type'] = poi_type  
          
        # 只保留通用列和'type'列  
        poi_data = poi_data[common_columns + ['type']]  
          
        # 如果是第一个文件，则直接将其赋值给all_poi_data  
        if all_poi_data is None:  
            all_poi_data = poi_data  
        else:  
            # 否则，将新的POI数据追加到all_poi_data中  
            all_poi_data = pd.concat([all_poi_data, poi_data], ignore_index=True)  
      
    # 将结果转换为GeoDataFrame  
    all_poi_data = gpd.GeoDataFrame(all_poi_data, geometry='geometry').to_crs('EPSG:32649')  
      
    return all_poi_data

def load_all_pois():  
    # 加载POI数据（不包括地铁）  
    poi_data = load_poi_data()    
      
    # 加载地铁数据  
    subway_data = load_subway()  
    # 给地铁数据添加一个'type'列，标记为'地铁'  
    subway_data['type'] = '交通'  
      
    # 将地铁数据添加到POI数据中  
    all_pois = pd.concat([poi_data, subway_data], ignore_index=True)  
      
    # 转换为GeoDataFrame  
    all_pois = gpd.GeoDataFrame(all_pois, geometry='geometry')  
      
    return all_pois    

@st.cache_data
def load_community_data():
    return gpd.read_file(gis_path+"//communities.shp").to_crs('EPSG:32649') 

def find_nearest_pois(community_point, type, poi_data):  
    # 初始化一个字典来存储结果  
    nearest_pois = {}  
    i = 1 
    
    # 确保使用投影坐标系统计算距离
    for poi_type in type:
        if i > 3:
            break  
        # 过滤出指定类型的POI  
        type_pois = poi_data[poi_data['type'] == poi_type].copy()  # 使用copy避免SettingWithCopyWarning
          
        # 计算每个POI与社区点的距离（单位：米）
        # 由于数据已经在EPSG:32649系统下，可以直接计算欧氏距离
        type_pois['distance'] = type_pois.geometry.distance(community_point.geometry)
        
        # 筛选2000米范围内的POI
        type_pois_within_2km = type_pois[type_pois['distance'] <= 2000]
        
        # 将距离四舍五入到整数
        type_pois_within_2km['distance'] = type_pois_within_2km['distance'].round(0)
          
        # 将结果添加到字典中  
        nearest_pois[poi_type] = type_pois_within_2km
        i += 1
      
    # 从地铁POI中找出最近的地铁  
    if '交通' in type and not nearest_pois['交通'].empty:  
        nearest_subway = nearest_pois['交通'].sort_values(by='distance').head(1)  
        nearest_subway_info = nearest_subway.to_dict(orient='records')[0]  
    else:  
        nearest_subway_info = {'name': '无地铁信息', 'distance': None}  

    return nearest_pois, nearest_subway_info 

def run(): 
    st.title("小区周边查询") 
    communities = load_community_data()
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