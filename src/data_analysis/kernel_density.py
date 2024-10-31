import geopandas as gpd  
import numpy as np  
from scipy.stats import gaussian_kde  
from osgeo import gdal, osr  
from pathlib import Path  
file_path = str(Path(__file__).resolve().parent.parent.parent / "data" / "gis_data") 
  
class KernelDensityEstimator:  
    def __init__(self, points_shp, polygon_shp, output_tiff, cell_size=1000):  
        """  
        初始化核密度估计器  
        :param points_shp: 点数据Shapefile路径  
        :param polygon_shp: 边界多边形Shapefile路径  
        :param output_tiff: 输出TIFF文件路径  
        :param cell_size: 输出栅格大小  
        :param bandwidth: 核密度带宽  
        """  
        self.points_shp = points_shp  
        self.polygon_shp = polygon_shp  
        self.output_tiff = output_tiff  
        self.cell_size = cell_size    
        self.points = None  
        self.polygon = None  
  
    def load_data(self):  
        """  
        加载数据  
        """  
        self.points = gpd.read_file(self.points_shp)  
        self.polygon = gpd.read_file(self.polygon_shp)
        self.points = self.points.to_crs("EPSG:3857")  
        self.polygon = self.polygon.to_crs("EPSG:3857") 

    def silverman_bw(self, data):  
        """  
        使用Silverman的经验法则计算带宽。  
        data: 输入数据，一个二维数组。  
        返回: 计算得到的带宽。  
        """  
        std_devs = np.std(data, axis=0)  
        n = data.shape[0]  
        return np.power(n, -0.2) * np.min(std_devs)   
  
    def compute_kernel_density(self):  
        """  
        计算核密度  
        """  
        # 提取点的坐标  
        points = np.vstack(self.points.geometry.apply(lambda x: (x.x, x.y)).values) 
        bandwidth = self.silverman_bw(points) 
        kde = gaussian_kde(points.T, bw_method=bandwidth)  
          
        # 创建栅格
        bounds = self.polygon.geometry.bounds
        xmin = min(bounds['minx'])
        ymin = min(bounds['miny'])
        xmax = max(bounds['maxx'])
        ymax = max(bounds['maxy']) 
          
        x = np.arange(xmin, xmax, self.cell_size)  
        y = np.arange(ymin, ymax, self.cell_size)  
        X, Y = np.meshgrid(x, y)  
        coordinates = np.vstack([X.ravel(), Y.ravel()])  
        Z = kde(coordinates).reshape(X.shape)  
          
        return X, Y, Z  
  
    def save_to_tiff(self, X, Y, Z):  
        """  
        保存核密度结果为TIFF文件  
        """  
        # 创建TIFF文件  
        driver = gdal.GetDriverByName('GTiff')  
        dst_ds = driver.Create(self.output_tiff, X.shape[1], X.shape[0], 1, gdal.GDT_Float32)  
          
        # 设置地理变换和投影  
        dst_ds.SetGeoTransform((X.min(), self.cell_size, 0, Y.max(), 0, -self.cell_size))  
        srs = osr.SpatialReference()  
        srs.ImportFromEPSG(4326)  # 假设使用WGS84坐标系统  
        dst_ds.SetProjection(srs.ExportToWkt())  
          
        # 写入数据  
        dst_ds.GetRasterBand(1).WriteArray(Z)  
        dst_ds.FlushCache()  
  
    def run(self):  
        """  
        执行核密度分析  
        """  
        self.load_data()  
        X, Y, Z = self.compute_kernel_density()  
        self.save_to_tiff(X, Y, Z)  
  
# 使用示例
points = file_path + "//communities.shp"
polygon = file_path + "//上海市_省界.shp"
kde = KernelDensityEstimator(points, polygon, 'output.tiff')  
kde.run()