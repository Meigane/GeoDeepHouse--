import streamlit as st  
import pandas as pd  
import geopandas as gpd  
from shapely.geometry import Point, Polygon  
import folium  
import sqlite3
from streamlit_folium import st_folium
from pathlib import Path
import json  
file_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" )
gis_path = file_path + "//gis_data"
i = 1

@st.cache_data
def load_community_data():
    return gpd.read_file(gis_path+"//communities.shp")

@st.cache_data
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
    gdf = gpd.GeoDataFrame(rows, geometry='geometry')  
  
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
    all_poi_data = gpd.GeoDataFrame(all_poi_data, geometry='geometry')  
      
    return all_poi_data  

@st.cache_data  
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

@st.cache_data
def load_all_pois():  
    # 加载POI数据（不包括地铁）  
    poi_data = load_poi_data()    
      
    # 加载地铁数据  
    subway_data = load_subway()  
    # 给地铁数据添加一个'type'列，标记为'地铁'  
    subway_data['type'] = '地铁'  
      
    # 将地铁数据添加到POI数据中  
    all_pois = pd.concat([poi_data, subway_data], ignore_index=True)  
      
    # 转换为GeoDataFrame  
    all_pois = gpd.GeoDataFrame(all_pois, geometry='geometry')  
      
    return all_pois  


  
def run():
    # all_pois = load_all_pois()
    # subway = load_subway()  
    # 加载数据  
    communities = load_community_data()  
    # poi_data = load_poi_data()  # 加载全上海市的POI数据  
    transaction_data = load_transaction_data()[0]  # 加载交易记录数据，假设返回的是一个元组，第一个元素是DataFrame  
      
    # 计算近五年成交数  
    recent_transactions = transaction_data[transaction_data['挂牌时间'] >= '2018-01-01']  
    community_transactions_count = recent_transactions.groupby('小区')['链家编号'].count().reset_index(name='近五年成交数')  
    communities = communities.merge(community_transactions_count, right_on='小区', left_on='name',how='left').fillna(0)
    communities = communities[communities['小区']!=0]  
    
    # 提取经纬度信息  
    communities['lat'] = communities['geometry'].apply(lambda x: x.y if x else None)  
    communities['lon'] = communities['geometry'].apply(lambda x: x.x if x else None)  
  
    # 过滤掉没有经纬度信息的行  
    communities = communities.dropna(subset=['lat', 'lon'])  
  
    # 创建用户表单  
    st.title("小区查询")
    filtered_communities = None
    dict = {'浦东':'310115','闵行':'310112','宝山':'310113','徐汇':'310104','普陀':'310107','杨浦':'310110',
        '长宁':'310105','松江':'310117','嘉定':'310114','黄浦':'310101','静安':'310106','虹口':'310109',
        '青浦':'310118','奉贤':'310120','金山':'310116','崇明':'310151'}
    with st.form(key="my_form"):  
        st.subheader("基于属性表达式查询记录")  
        district = st.selectbox("选择区：", dict.keys())  
        min_price = st.number_input("最低单价：", min_value=0, value=70000)  
        max_price = st.number_input("最高单价：", min_value=0, value=100000)  
        submitted1 = st.form_submit_button('提交')  

        if submitted1:    
            # 根据用户输入过滤数据  
            filtered_communities = communities[  
                (communities['区'] == district) &  
                (communities['price'] >= min_price) &  
                (communities['price'] <= max_price)  
            ]
            # 创建folium地图  
            if not filtered_communities.empty:  
                map_location = [filtered_communities['lat'].mean(), filtered_communities['lon'].mean()]  
                m = folium.Map(location=map_location, zoom_start=11)
                folium.TileLayer(tiles = 'Gaode.Normal',name="高德地图",control=True).add_to(m)  
                #folium.TileLayer(tiles = 'CartoDB.Positron',name="Carto地图").add_to(m)
                features = []
                # 在地图上添加房价分布的圆形标记  
                for index, row in filtered_communities.iterrows(): 
                    # 保留两位小数  
                    price_str = f"{round(row['price'], 2):.2f}"  
                    
                    # 成交数转换为整数  
                    transaction_count = int(row.get('近五年成交数', 0))  
                    popup_text = f"<h5>{row['name']}</h5><p>均价：{price_str}元/平米<br>近五年成交数：{transaction_count}</p>"
                    feature = folium.CircleMarker(  
                        location=[row['lat'], row['lon']],  
                        radius=5,  
                        popup=popup_text,  
                        color='crimson',  
                        fill=True,  
                        fill_color='crimson'  
                    )
                    feature.add_to(m) 
                    features.append(feature)

            
                st_folium(m, width=700, height=500)
            else:  
                st.write("没有找到符合条件的小区。") 


if __name__ == "__main__":  
    run()