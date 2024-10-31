import geopandas as gpd  
import numpy as np  
from scipy.interpolate import griddata  
import rasterio  
from rasterio.transform import from_origin  
from shapely.geometry import box  
from osgeo import ogr, gdal, osr  
import numpy as np
from pathlib import Path
from pykrige.ok import OrdinaryKriging 
from scipy.spatial import distance 
from rasterio.transform import from_origin  
from tqdm import tqdm  
  
module_path = Path(__file__).resolve().parent.parent / 'data_collection'  
file_path = str(Path(__file__).resolve().parent.parent.parent / "data" / "gis_data")  
  
class Interpolator:  
    def __init__(self, points_shp, polygon_shp, field_to_interpolate, grid_size=50):  
        self.points = gpd.read_file(points_shp)  
        self.polygon = gpd.read_file(polygon_shp)  
        self.field_to_interpolate = field_to_interpolate  
        self.grid_size = grid_size  
  
    def prepare_grid(self):  
        xmin, ymin, xmax, ymax = self.polygon.geometry.total_bounds  
        self.grid_x, self.grid_y = np.mgrid[xmin:xmax:self.grid_size*1j, ymin:ymax:self.grid_size*1j]  
  
    def interpolate(self):  
        points_xy = self.points['geometry'].apply(lambda geom: (geom.x, geom.y))
        points_xy = np.array(list(points_xy))    
        z = self.points[self.field_to_interpolate].values 
                # 设置克里金插值的参数  
        # OK = OrdinaryKriging(  
        #     x=points_xy[:,0],
        #     y=points_xy[:,1],  # 散点坐标  
        #     z=z,     # 散点值  
        #     variogram_model='linear',  # 变差函数模型，可以是'linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect'  
        #     verbose=False,  
        #     enable_plotting=False,  
        # )  
        
        # # 定义插值网格  
        # yres = (max(points_xy[:, 1]) - min(points_xy[:, 1])) / self.grid_size
        # xres = (max(points_xy[:, 0]) - min(points_xy[:, 0])) / self.grid_size  
        # grid_lon = np.arange(min(points_xy[:, 0]), max(points_xy[:, 0]), xres)  
        # grid_lat = np.arange(min(points_xy[:, 1]), max(points_xy[:, 1]), yres)  
        
        # # 执行克里金插值  
        # self.grid_z, ss_interp = OK.execute('grid', grid_lon, grid_lat)   
        # self.grid_z = griddata(points_xy, z, (self.grid_x, self.grid_y), method='cubic') 
        # return self.grid_z
        xi, yi = self.grid_x.flatten(), self.grid_y.flatten() 
        grid_z = np.array([self.IDW(points_xy[:, 0], points_xy[:, 1], z, [x], [y])[0][2] for x, y in zip(xi, yi)])  
        self.grid_z = grid_z.reshape(self.grid_x.shape)  
        return self.grid_z  
  
    def IDW(self, x, y, z, xi, yi):
        # 转换为numpy数组  
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)  
        xi, yi = np.asarray(xi), np.asarray(yi)  
        
        # 计算所有预测点到所有已知点的距离  
        dists = distance.cdist(list(zip(xi, yi)), list(zip(x, y)), 'euclidean')  
        
        # 避免除以零  
        dists[dists == 0] = 1e-10  
        
        # 计算权重和加权和  
        weights = 1 / np.power(dists, 2)  
        sumsup = np.dot(weights, z)  
        suminf = np.sum(weights, axis=1)  
        
        # 计算插值结果  
        u = sumsup / suminf  
        
        # 返回结果  
        return list(zip(xi, yi, u))  
  
    def save_to_tiff(self, output_file):  
        # 获取输入面要素的CRS  
        crs = self.polygon.crs  
        
        # 计算地理范围  
        xmin, ymin, xmax, ymax = self.polygon.geometry.total_bounds  
        
        # 计算像素大小和行数、列数
        yres = (ymax - ymin) / self.grid_size
        xres = (xmax - xmin) / self.grid_size  
        
        # 计算transform参数  
        transform = rasterio.transform.from_origin(xmin, ymin, xres, -yres)  # 注意y方向是向下的  
        
        # 打开TIFF文件并写入数据  
        with rasterio.open(  
            output_file, 'w', driver='GTiff',  
            height=self.grid_size, width=self.grid_size,  
            count=1, dtype=self.grid_z.dtype, crs=crs,  
            transform=transform  
        ) as dst:  
            dst.write(self.grid_z, 1)
  
if __name__ == "__main__":
    interpolator = Interpolator(  
        points_shp=file_path + "//communities.shp",  
        polygon_shp=file_path + "//上海市_省界.shp",  
        field_to_interpolate="price"  
    )  
    interpolator.prepare_grid()  
    interpolator.interpolate()  
    interpolator.save_to_tiff(file_path + "//interpolated_output.tif")