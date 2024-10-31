import geopandas as gpd  
import pandas as pd  
import requests 
import math 
from shapely.geometry import Point, shape  
import sys  
from pathlib import Path  
  
module_path = Path(__file__).resolve().parent.parent / 'data_collection'  
file_path = str(Path(__file__).resolve().parent.parent.parent / "data" / "gis_data")
sys.path.append(str(module_path))  
from sql_saver import SqlSaver
from get_location import BaiduMapCoordConverter
  
class GeoProcessor(BaiduMapCoordConverter):  
    def __init__(self): 
        super().__init__() 
        self.db = SqlSaver("GisProject", "processed")
        self.boundaries = None

    def load_boundaries(self):  
        # 加载行政区边界数据  
        boundaries = gpd.read_file(file_path + '//boundaries.shp')  
        return boundaries  
  
    def save_points(self, query, output):  
        df = pd.read_sql_query(query, self.db.engine)  
          
        # 创建GeoDataFrame  
        gdf = gpd.GeoDataFrame(  
            df,  
            geometry=gpd.points_from_xy(df.x, df.y)  
        )  
          
        gdf.set_crs(epsg=4326, inplace=True)  
          
        # 判断点所在的行政区 
        self.boundaries = self.load_boundaries() 
        gdf['区'] = gdf['geometry'].apply(lambda point: self.find_district(point))  
          
        # 保存为Shapefile  
        gdf.to_file(file_path + f'//{output}.shp', driver='ESRI Shapefile', encoding='utf-8')  
  
    def find_district(self, point):  
        # 判断点在哪个行政区内  
        for _, row in self.boundaries.iterrows():  
            if row['geometry'].contains(point):  
                return row['name']  
        return None  # 如果点不在任何行政区内，则返回None 
     
    # def save_points(self, query, output): 
    #     df = pd.read_sql_query(query, self.db.engine)  
        
    #     # 创建GeoDataFrame  
    #     gdf = gpd.GeoDataFrame(  
    #         df,  
    #         geometry=gpd.points_from_xy(df.x, df.y)  
    #     )  
        
    #     gdf.set_crs(epsg=4326, inplace=True)  
    #     gdf.to_file(file_path + f'//{output}.shp', driver='ESRI Shapefile',encoding='utf-8')

    def save_boundry(self):
        all_boundaries =  gpd.GeoDataFrame()
        dict = {'浦东':'310115','闵行':'310112','宝山':'310113','徐汇':'310104','普陀':'310107','杨浦':'310110',
        '长宁':'310105','松江':'310117','嘉定':'310114','黄浦':'310101','静安':'310106','虹口':'310109',
        '青浦':'310118','奉贤':'310120','金山':'310116','崇明':'310151'}#创建一个区名和adcode对应的索引字典
        for d, adcode in dict.items(): 
            url = 'https://geo.datav.aliyun.com/areas_v3/bound/'+adcode+'.json'
            response = requests.get(url)
            geojson = response.json()#获取geojson数据
            for i in range(len(geojson['features'][0]['geometry']['coordinates'][0][0])):#进行坐标转换到WGS84
                geojson['features'][0]['geometry']['coordinates'][0][0][i] = self.gcj02_to_wgs84(geojson['features'][0]['geometry']['coordinates'][0][0][i][0],geojson['features'][0]['geometry']['coordinates'][0][0][i][1])
            bound = shape(geojson['features'][0]['geometry']) 

            boundary_gdf = gpd.GeoDataFrame({'geometry': [bound], 'name': [d]})  
            all_boundaries = pd.concat([all_boundaries, boundary_gdf], ignore_index=True)

        # 保存为Shapefile  
        all_boundaries.to_file(file_path + '//boundaries.shp', driver='ESRI Shapefile', encoding='utf-8')


if __name__ == "__main__": 
     
    processor = GeoProcessor()
    query = """SELECT name, x, y, price, cnt FROM communities,
                (SELECT 小区, AVG(单价) as price, count(*) as cnt FROM house_info GROUP BY 小区) AS p
                WHERE name=小区 AND x IS NOT NULL;""" 
     
    processor.save_boundry() 
    processor.save_points(query, "communities")