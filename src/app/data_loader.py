import streamlit as st  
import pandas as pd  
import folium   
from pathlib import Path  
file_path = str(Path(__file__).resolve().parent.parent.parent / "data" )
gis_path = file_path + "//gis_data"
import geopandas as gpd 
import sqlite3

def load_community_data():
    return gpd.read_file(gis_path+"//communities.shp")
  
def load_poi_data():  
    poi_files = [  
        "餐饮服务.shp",  
        "风景名胜.shp",  
        "购物服务.shp",  
        "科教文化.shp",  
        "生活服务.shp",  
        "体育休闲.shp",  
        "医疗保健.shp"  
    ]  
      
    all_poi_data = None  
    common_columns = ['geometry']  # 保留的通用列，至少包含geometry  
      
    for file_name in poi_files:  
        poi_data = gpd.read_file(gis_path + "//" + file_name)  
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
    all_poi_data = gpd.GeoDataFrame(all_poi_data, geometry='geometry')  
      
    return all_poi_data  
  
def load_transaction_data(): 
    db_path = file_path + "//processed//GisProject.db"
    # 连接到SQLite数据库  
    conn = sqlite3.connect(db_path)  
    dic = {'小区': 'TEXT', '房型': 'TEXT', '面积': 'REAL',
            '房屋户型': 'TEXT', '所在楼层': 'TEXT', '户型结构': 'TEXT', 
            '建筑类型': 'TEXT', '房屋朝向': 'TEXT', '建成年代': 'INTEGER', 
            '装修情况': 'TEXT', '建筑结构': 'TEXT', '梯户比例': 'TEXT',
            '配备电梯': 'TEXT', '链家编号': 'TEXT', '交易权属': 'TEXT', 
            '挂牌时间': 'DATE', '房屋用途': 'TEXT', '房权所属': 'TEXT',
            '总价': 'FLOAT', '单价': 'FLOAT', '楼层数': 'INTEGER',
            '所在地区': 'TEXT', '室数': 'INTEGER', '厅数': 'INTEGER'}  
    # 执行SQL查询以加载交易记录数据  
    query = """  
    SELECT * FROM house_info  
    """  
    
    transaction_data = pd.read_sql_query(query, conn)  
      
    # 关闭数据库连接  
    conn.close()  
      
    return transaction_data, dic