import sys  
from pathlib import Path  
  
module_path = Path(__file__).resolve().parent.parent / 'data_collection'  
sys.path.append(str(module_path))  

from sql_saver import SqlSaver
# from data_collection.sql_saver import SqlSaver
import sqlite3
import pandas as pd

raw_db = SqlSaver("GisProject", "raw")
new_db = SqlSaver("GisProject", "processed")

district = {  
    'beicai': '北蔡',  
    'biyun': '碧云',  
    'caolu': '曹路',  
    'chuansha': '川沙',  
    'datuanzhen': '大团镇',  
    'gaodong': '高东',  
    'gaohang': '高行',  
    'geqing': '高青',  
    'hangtou': '航头',  
    'huamu': '花木',  
    'huinan': '惠南',  
    'jinqiao': '金桥',  
    'jinyang': '金杨',  
    'kangqiao': '康桥',  
    'laogangzhen': '老港镇',  
    'lianyang': '联洋',  
    'lingangxincheng': '临港新城',  
    'lujiazui': '陆家嘴',  
    'meiyuan1': '梅园',  
    'nanmatou': '南码头',  
    'nichengzhen': '泥城镇',  
    'sanlin': '三林',  
    'shibo': '世博',  
    'shuyuanzhen': '书院镇',  
    'tangqiao': '塘桥',  
    'tangzhen': '唐镇',  
    'waigaoqiao': '外高桥',  
    'wanxiangzhen': '万祥镇',  
    'weifang': '潍坊',  
    'xinchang': '新场',  
    'xuanqiao': '宣桥',  
    'yangdong': '杨东',  
    'yangjing': '杨泾',  
    'yangsiqiantan': '杨思前滩',  
    'qiantan': '前滩',  
    'yuanshen': '源深',  
    'yuqiao1': '御桥',  
    'zhangjiang': '张江',  
    'zhoupu': '周浦',  
    'zhuqiao': '祝桥'  
}

# 进行数据预处理
for key, value in district.items():
    sql = [f"ALTER TABLE {key}_info DROP COLUMN 成交时间;",
           f"ALTER TABLE {key}_info DROP COLUMN 供暖方式;",
           f"ALTER TABLE {key}_info DROP COLUMN 房屋年限;",
           f"ALTER TABLE {key}_info ADD COLUMN 楼层类型 TEXT;" , 
           f"ALTER TABLE {key}_info ADD COLUMN 楼层数 INTEGER;",
           f"ALTER TABLE {key}_info ADD COLUMN 所在地区 TEXT;",
           f"UPDATE {key}_info " +  
           f"""SET 
                楼层类型 = SUBSTR(所在楼层, 1, 
                                INSTR(所在楼层, '（') - 1),  
                楼层数 = CAST(SUBSTR(所在楼层, INSTR(所在楼层, '共') + 1, 
                            INSTR(所在楼层, '层') - 1) AS INTEGER),
                所在地区 = "{value}";"""]

    for s in sql:
        with sqlite3.connect(raw_db.db_path) as conn:
            conn.execute(s)

# -----------------------------------------------------------------------------
# 测试用
# key = "beicai"
# sql = [f"ALTER TABLE {key}_info DROP COLUMN 成交时间;",
#         f"ALTER TABLE {key}_info DROP COLUMN 供暖方式;",
#         f"ALTER TABLE {key}_info DROP COLUMN 房屋年限;"]

# for s in sql:
#     with sqlite3.connect(raw_db.db_path) as conn:
#         conn.execute(s)

# community_dic = {"name":"TEXT PRIMARY KEY",
#                  "x":"FLOAT", "y":"FLOAT"}

# new_db.create_table("communities", community_dic)

# for key in district:
#     rows = raw_db.show_data(sql=f"SELECT DISTINCT 小区 FROM {key}_info;")
#     if rows:
#         data = [row[0] for row in rows]
#         df = pd.DataFrame({'name': data})
#         new_db.insert_data("communities", df)

# print(new_db.show_data(sql="SELECT COUNT(*) FROM communities;"))

# print(new_db.show_data(sql="SELECT * FROM communities LIMIT 25;"))
# print(new_db.show_data(sql="SELECT AVG(建成年代) FROM house_info;"))
# print(new_db.show_data(sql="SELECT DISTINCT 配备电梯 FROM house_info;"))
# -----------------------------------------------------------------------------

rows = raw_db.show_data(sql="PRAGMA table_info(beicai_info);")
house_dic = {row[1] : row[2] for row in rows}
new_db.create_table("house_info", house_dic)

# 读取源数据库中的数据并插入新数据库
for key in district.keys():
    rows = raw_db.show_data(table_name=f"{key}_info")   
    
    with sqlite3.connect(new_db.db_path) as conn:
        conn.executemany("""INSERT INTO "house_info" 
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)

# 填补空缺值  
set_sql = [  
    "ALTER TABLE house_info DROP COLUMN 楼层类型;",  
    "UPDATE house_info SET 户型结构='平层' WHERE 户型结构 IS NULL;",  
    "UPDATE house_info SET 建筑类型='板楼' WHERE 建筑类型 IS NULL;",  
    "UPDATE house_info SET 房屋朝向='无' WHERE 房屋朝向 IS NULL;",  
    "UPDATE house_info SET 建成年代=2001 WHERE 建成年代 IS NULL;",  
    "UPDATE house_info SET 配备电梯='无' WHERE 配备电梯 IS NULL;",  
    "UPDATE house_info SET 梯户比例='一梯两户' WHERE 梯户比例 IS NULL AND 配备电梯='有';",  
    "UPDATE house_info SET 梯户比例='无电梯' WHERE 梯户比例 IS NULL AND 配备电梯='无';"  
]  

for sql in set_sql:  
    new_db.show_data(sql=sql, no_return=True)  

rows = new_db.show_data(sql="PRAGMA table_info(house_info);")
house_info_dic = {row[1] : row[2] for row in rows}
rows = new_db.show_data(sql="PRAGMA table_info(communities);")
com_info_dic = {row[1] : row[2] for row in rows}
print(1)

# 删除空值
for key in house_info_dic.keys():
    sql = f"""DELETE FROM house_info
            WHERE {key} IS NULL 
                    OR {key} = '';"""
    new_db.show_data(sql=sql, no_return=True)

for key in house_info_dic.keys():
    sql = f"""SELECT count(*) 
            FROM house_info
            where {key} is NULL 
                    or {key} = '';"""
    print(key, "共有", new_db.show_data(sql=sql)[0][0], "个空值")

print(new_db.show_data(sql="SELECT COUNT(*) FROM house_info;"))

#创建小区表，为地理编码做准备
new_db.create_table("communities", {'name':"TEXT PRIMARY KEY", "x":"FLOAT", "y":"FLOAT"})
    
rows = new_db.show_data(sql="SELECT DISTINCT 小区 FROM house_info;")
with sqlite3.connect(new_db.db_path) as conn:
    conn.executemany("""INSERT INTO "communities" (name) 
                        VALUES (?)""", rows)
    
print(new_db.show_data(sql="SELECT COUNT(*) FROM communities;"))
set_sql = ["ALTER TABLE house_info ADD COLUMN 室数 INTEGER;",
       "ALTER TABLE house_info ADD COLUMN 厅数 INTEGER;",
        """  
        UPDATE house_info    
        SET 室数 = CAST(SUBSTR(房屋户型, 1, INSTR(房屋户型, '室') - 1) AS INTEGER),  
            厅数 = CAST(SUBSTR(房屋户型, INSTR(房屋户型, '室') + LENGTH('室'), INSTR(房屋户型, '厅') - INSTR(房屋户型, '室') - LENGTH('室')) AS INTEGER);  
        """]
for sql in set_sql:  
    new_db.show_data(sql=sql, no_return=True)

print(new_db.show_data(sql="SELECT 房屋户型, 室数, 厅数 FROM house_info LIMIT 25;"))