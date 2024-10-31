import folium
from folium.plugins import HeatMap
import geopandas as gpd
from pathlib import Path
from streamlit_folium import st_folium
import streamlit as st
import branca
from folium.features import GeoJson
from folium import Marker, FeatureGroup   

file_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" /"gis_data")
gdf = gpd.GeoDataFrame.from_file(file_path + "//communities.shp")


  
# 读取数据  
boundary = gpd.read_file(file_path + '//boundaries.shp')  
communities = gpd.read_file(file_path + '//communities.shp')
communities_projected = communities.to_crs("EPSG:4326")  
#communities.crs = "EPSG:4326"
communities['area'] = communities.geometry.area  
  
# 计算每个区的cnt总和以及cnt密度（sum(cnt)除以面积）  
cnt_sum = communities.groupby('区')['cnt'].sum().reset_index()  
  
# 合并cnt总和和面积数据  
cnt_sum.columns = ['DISTRICT', 'CNT_SUM']  
  
# 将cnt总和合并到boundary数据  
boundary = boundary.merge(cnt_sum, left_on='name', right_on='DISTRICT', how='left')  
  
# 创建一个colormap函数  
minimum, maximum = cnt_sum["CNT_SUM"].quantile([0.05, 0.75])
colormap = branca.colormap.LinearColormap(
    colors=["yellow", "red"],
    vmin=round(minimum,2),
    vmax=round(maximum,2))
 
  
# 创建地图  
m = folium.Map(location=[communities['y'].mean(), communities['x'].mean()], zoom_start=9)  
  
# 定义style_function  
def style_function(feature):  
    style = {  
        "fillColor": colormap(feature["properties"]["CNT_SUM"]),  
        "color": "black",  
        "weight": 2,  
        "fillOpacity": 0.9,  
    }  
    return style  
  
# 将GeoJson添加到地图 
boundary.crs = "EPSG:4326"
point_layer = FeatureGroup(name='出售房屋数量', control=True) 
GeoJson(data=boundary,  
        style_function=style_function).add_to(point_layer)  
point_layer.add_to(m)

c_list =[] 
for i in range(len(gdf)):
    record = gdf.iloc[i]
    lng = float(record["geometry"].x)
    lat = float(record["geometry"].y)
    weight = float(record["price"])
    c_list.append([lat,lng,weight])

heat_layer = FeatureGroup(name='房价热力图', control=True) 
HeatMap(c_list,radius=15,blur=10).add_to(heat_layer)
heat_layer.add_to(m)

folium.TileLayer(tiles = 'Gaode.Normal',name="高德地图",control=True).add_to(m)

folium.LayerControl().add_to(m)
st.title("房价热点图")
st_folium(m, width=700, height=500)
