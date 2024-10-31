import pandas as pd  
import sqlite3  
from sklearn.preprocessing import LabelEncoder 
from pathlib import Path
import sys 
import joblib 
module_path = Path(__file__).resolve().parent.parent / 'data_collection' 
file_path = str(Path(__file__).resolve().parent.parent.parent )
sys.path.append(str(module_path))  

from sql_saver import SqlSaver  

conn = sqlite3.connect(str(file_path) + '\\data\\processed\\' + 'GisProject.db')  
  
# 读取house_info表  
query = "SELECT * FROM house_info;"  
df = pd.read_sql_query(query, conn)  
  
# 关闭数据库连接  
conn.close()  
  
# 文本列名  
text_columns = ['房型', '房屋户型', '所在楼层', '户型结构', '建筑类型', '房屋朝向',   
                '装修情况', '建筑结构', '梯户比例', '配备电梯', '交易权属',   
                '房屋用途', '房权所属', '所在地区']  
# 保存LabelEncoder映射的路径  
encoder_path =file_path+'//models//encoders.joblib' 
   
encoders = {}  
  
# 对文本数据进行预处理并保存LabelEncoder  
for col in text_columns:  
    # 去除前后空格  
    df[col] = df[col].str.strip()  
    # 如果LabelEncoder已经存在，则使用它；否则，创建一个新的  
    if col not in encoders:  
        encoders[col] = LabelEncoder()  
    df[col] = encoders[col].fit_transform(df[col])  

joblib.dump(encoders, encoder_path)  
  
# 处理日期数据  
df['挂牌时间'] = pd.to_datetime(df['挂牌时间'])  
# 可以将日期转换为距离某个特定日期的天数或月份数，例如转换为距离最晚日期的天数  
max_date = df['挂牌时间'].max()  
df['挂牌时间'] = (max_date - df['挂牌时间']).dt.days

df = df.drop(['小区', '链家编号'], axis=1)
  
db = SqlSaver("GisProject", "processed")
dic = {'房型': 'INTEGER', '面积': 'REAL', 
       '房屋户型': 'INTEGER', '所在楼层': 'INTEGER', '户型结构': 'INTEGER', 
       '建筑类型': 'INTEGER', '房屋朝向': 'INTEGER', '建成年代': 'INTEGER', 
       '装修情况': 'INTEGER', '建筑结构': 'INTEGER', '梯户比例': 'INTEGER', '配备电梯': 'INTEGER', 
       '交易权属': 'INTEGER', '挂牌时间': 'DATE', '房屋用途': 'INTEGER', 
       '房权所属': 'INTEGER', '总价': 'FLOAT', '单价': 'FLOAT', '楼层数': 'INTEGER', '所在地区': 'INTEGER'}
db.show_data(sql="DROP TABLE house_label;")
db.create_table("house_label", dic)
db.insert_data("house_label",df)