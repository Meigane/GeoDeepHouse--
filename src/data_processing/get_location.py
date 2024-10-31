import sqlite3  
import requests  
import json  
import hashlib  
from urllib.parse import quote_plus, quote  
import math
from tqdm import tqdm 

import sys  
from pathlib import Path  
  
module_path = Path(__file__).resolve().parent.parent / 'data_collection'  
sys.path.append(str(module_path))  

from sql_saver import SqlSaver
# from data_collection.sql_saver import SqlSaver
  
class BaiduMapCoordConverter:  
    def __init__(self, ak=None):  
        self.db = SqlSaver("GisProject","processed")
        self.ak = ak
        self.fail = []  
          
        # 坐标转换常量  
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0  
        self.pi = 3.1415926535897932384626  # π  
        self.a = 6378245.0  # 长半轴  
        self.ee = 0.00669342162296594323  # 偏心率平方  
          
    def get_community_names_from_db(self):  
        # 连接到SQLite数据库  
        rows = self.db.show_data(sql="""SELECT DISTINCT name 
                                        FROM communities
                                        WHERE x is NULL;""")  
        community_names = [row[0] for row in rows]  

        return community_names  
      
    def get_baidu_coords(self, address):
        address = "上海" + address  
        # 百度地图API的URL模板  
        url_template = "http://api.map.baidu.com/geocoding/v3/?address={}&output=json&ak={}"  
        url = url_template.format(quote(address), self.ak)  
          
        # 发送HTTP请求并获取响应
        try:  
            response = requests.get(url)
        except:
            self.fail.append(address)
            return None, None 
        data = response.json()  
          
        # 提取经纬度（假设查询成功）  
        if data['status'] == 0:  
            location = data['result']['location']  
            return location['lng'], location['lat']  
        else:  
            return None, None   
  
    def bd09_to_wgs84(self, bd_lon, bd_lat):  
        """  
        将百度坐标（BD-09）转换为WGS84坐标  
        """  
        lon, lat = self.bd09_to_gcj02(bd_lon, bd_lat)  
        return self.gcj02_to_wgs84(lon, lat)  
      
    def bd09_to_gcj02(self, bd_lon, bd_lat):  
        """  
        百度坐标（BD-09）转火星坐标（GCJ-02）  
        火星坐标系GCJ-02是由中国国家测绘局制订的地理信息系统的坐标系统。  
        由经纬度B转换到经纬度G的转换公式：  
        """  
        x = bd_lon - 0.0065  
        y = bd_lat - 0.006  
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)  
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)  
        gg_lng = z * math.cos(theta)  
        gg_lat = z * math.sin(theta)  
        return [gg_lng, gg_lat]  
      
    def gcj02_to_wgs84(self, lng, lat):  
        """  
        火星坐标系（GCJ-02）转WGS84  
        """  
        dlat = self._transformlat(lng - 105.0, lat - 35.0)  
        dlng = self._transformlng(lng - 105.0, lat - 35.0)  
        radlat = lat / 180.0 * self.pi  
        magic = math.sin(radlat)  
        magic = 1 - self.ee * magic * magic  
        sqrtmagic = math.sqrt(magic)  
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrtmagic) * self.pi)  
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)  
        mglat = lat + dlat  
        mglng = lng + dlng  
        return [mglng, mglat]  
      
    def _transformlat(self, lng, lat):  
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
            0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * self.pi) + 40.0 *
                math.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * self.pi) + 320 *
                math.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret  
      
    def _transformlng(self, lng, lat):  
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * self.pi) + 40.0 *
                math.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * self.pi) + 300.0 *
                math.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret  

    def get_wgs84_coords_for_communities(self):  
        community_names = self.get_community_names_from_db()  
          
        for name in tqdm(community_names):    
            bd_lng, bd_lat = self.get_baidu_coords(name)  
            if bd_lng is not None and bd_lat is not None:  
                wgs84_lon, wgs84_lat = self.bd09_to_wgs84(bd_lng, bd_lat)
                sql = f"""UPDATE communities
                            SET 
                            x = {wgs84_lon},  
                            y = {wgs84_lat}
                            WHERE name = "{name}";"""
                self.db.show_data(sql=sql, no_return=True)

        with open('fail.txt', 'a') as file:  
            for item in self.fail:  
                file.write("%s\n" % item)
        print("Finish!") 
             
  
 
if __name__ == "__main__":  
    converter = BaiduMapCoordConverter('RQmw2ZPtIa9qGK3BOXfWFcegCNuShJkq')  
    wgs84_coords = converter.get_wgs84_coords_for_communities()  