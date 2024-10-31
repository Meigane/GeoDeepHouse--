import sqlite3  
from sqlalchemy import create_engine  
import pandas as pd  
from sqlalchemy.exc import IntegrityError 
from pathlib import Path 
  
class SqlSaver:  
    DATA_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "gis_data") + "//"  
  
    def __init__(self, db_name, path):  
        self.db_path = self.DATA_PATH + path + "\\" + db_name + ".db"  
        self.engine = create_engine('sqlite:///' + self.db_path) 
  
    def create_table(self, table_name, dic):  
        sql_parts = [f"{key} {value}" for key, value in dic.items()]  
        attribute_statement = ', '.join(sql_parts)  
        sql_statements = f"""CREATE TABLE IF NOT EXISTS {table_name}  
                            ({attribute_statement});"""  
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql_statements)  

    def drop_table(self, table_name):
        sql = f"DROP TABLE IF EXISTS {table_name};"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql)
  
    def insert_data(self, table_name, data):  
        try:  
            data.to_sql(table_name, self.engine, index=False, if_exists='append')  
        except IntegrityError as e:  
            # 在这里处理或忽略异常  
            pass
  
    def show_data(self, table_name=None, sql=None, no_return=False):  
        with sqlite3.connect(self.db_path) as conn:
            if not sql:  
                sql = f"SELECT * FROM {table_name};"

            cursor = conn.execute(sql) 

            if no_return:
                return 
            
            info = cursor.fetchall()  
            return info  

if __name__ == "__main__":
    db = SqlSaver("GisProject", "processed")
    db.show_data(sql="SELECT 总价 FROM house_info LIMIT25;")
    dic = {"name":"TEXT PRIMARY KEY","x":"FLOAT","y":"FLOAT"}
    # db.create_table("communities",)
    DISTRICT_MAP = {
    "黄浦区": "huangpu",  "徐汇区": "xuhui",  
    "长宁区": "changning",  "静安区": "jingan",  
    "普陀区": "putuo",  "虹口区": "hongkou",  
    "杨浦区": "yangpu",  "闵行区": "minhang",  
    "宝山区": "baoshan",  "嘉定区": "jiading",  
    "浦东新区": "pudong",  "金山区": "jinshan",  
    "松江区": "songjiang",  "青浦区": "qingpu",  
    "奉贤区": "fengxian",  "崇明区": "chongming"  
    }

    DISTRICT = {"浦东新区":['beicai', 'biyun', 'caolu', 'chuansha', 
                        'datuanzhen', 'gaodong', 'gaohang', 'geqing',
                        'hangtou', 'huamu', 'huinan', 'jinqiao', 'jinyang', 
                        'kangqiao', 'laogangzhen', 'lianyang', 'lingangxincheng',
                        'lujiazui','meiyuan1', 'nanmatou', 'nichengzhen', 'sanlin', 'shibo', 
                        'shuyuanzhen','tangqiao', 'tangzhen', 'waigaoqiao', 
                        'wanxiangzhen', 'weifang', 'xinchang', 'xuanqiao', 
                        'yangdong', 'yangjing', 'yangsiqiantan', 'qiantan', 'yuanshen',
                        'yuqiao1', 'zhangjiang', 'zhoupu', 'zhuqiao'],
            "闵行区":['caohejing', 'chunshen', 'gumei', 'hanghua', 
                        'huacao', 'jinganxincheng', 'jinhongqiao', 'jinhui',
                        'laominxing', 'longbai', 'maqiao', 'meilong', 'minpu', 'pujiang1', 
                        'qibao', 'xinminbieshu', 'wujing', 'xinzhuangbeiguangchang', 
                        'xinzhuangnanguangchang', 'zhuanqiao']
            }
    url_dic = {"name": "TEXT", "url": "TEXT PRIMARY KEY"}
    house_info = {'小区': "TEXT", '房型': "TEXT", '面积':"REAL", '成交时间':"DATE", 
                  '房屋户型':"TEXT", '所在楼层':"TEXT", 
                  '户型结构':"TEXT", '建筑类型':"TEXT",
                  '房屋朝向':"TEXT", '建成年代':"INTEGER", '装修情况':"TEXT", 
                  '建筑结构':"TEXT", '供暖方式':"TEXT", '梯户比例':"TEXT", 
                  '配备电梯':'TEXT', '链家编号':"TEXT PRIMARY KEY", '交易权属':"TEXT", 
                  '挂牌时间':"DATE", '房屋用途':"TEXT", '房屋年限':"TEXT", 
                  '房权所属':"TEXT", '总价':"FLOAT", '单价':"FLOAT"}

    for district in DISTRICT_MAP.values():
        db.create_table(district, dic)
    print(db.show_data("chongming"))

    for district in DISTRICT_MAP.values():
        db.drop_table(district)

    for district in DISTRICT_MAP.values():
        db.drop_table(district+"_url")

    for district in DISTRICT["浦东新区"]:
        db.create_table(district+"_url", url_dic)

    for district in DISTRICT_MAP.values():
        db.drop_table(district+"_info")

    for district in DISTRICT["浦东新区"]:
        db.create_table(district+"_info", house_info)


    with sqlite3.connect(db.db_path) as conn:  
        cursor = conn.execute("""SELECT 小区, AVG(总价) AS 平均总价,
                               AVG(单价) AS 平均单价 
                              FROM chongming_info GROUP BY 小区;""")
    print(cursor.fetchall())
    conn.close()
    with sqlite3.connect(db.db_path) as conn:  
        cursor = conn.execute("""SELECT count(DISTINCT 小区)
                                FROM chongming_info;""")
    print(cursor.fetchall())

    for key in house_info.keys():
        sql = f"""SELECT count(*) 
                FROM beicai_info
                where {key} is NULL 
                        or {key} = '';"""
        print(key, "共有", db.show_data("beicai_info", sql)[0][0], "个空值")

    sql = """SELECT * 
             FROM beicai_info
             LIMIT 1;"""
    print(db.show_data(sql=sql))

    sql = ["ALTER TABLE beicai_info ADD COLUMN 楼层类型 TEXT;" , 
            "ALTER TABLE beicai_info ADD COLUMN 楼层数 INTEGER;",
             """UPDATE beicai_info 
             SET 
                楼层类型 = SUBSTR(所在楼层, 1, 
                                INSTR(所在楼层, '（') - 1),  
                楼层数 = CAST(SUBSTR(所在楼层, INSTR(所在楼层, '共') + 1, 
                            INSTR(所在楼层, '层')- 1) AS INTEGER);"""]
    
    with sqlite3.connect(db.db_path) as conn: 
            conn.execute(r"""UPDATE beicai_info  
                            SET 楼层数 = CAST(SUBSTR(所在楼层, INSTR(所在楼层, '共') + 1, 
                            INSTR(所在楼层, '层') - 1) AS INTEGER);""")

    sql = r"""SELECT 所在楼层,楼层类型,  楼层数
             FROM beicai_info
             WHERE 所在楼层 LIKE '%(共%层)%'
             LIMIT 15;"""
    print(db.show_data(sql=sql))
    
    for key in DISTRICT["浦东新区"]:
        sql = [f"ALTER TABLE {key}_info DROP COLUMN 成交时间;",
               f"ALTER TABLE {key}_info DROP COLUMN 供暖方式;",
               f"ALTER TABLE {key}_info DROP COLUMN 房屋年限;",
               f"ALTER TABLE {key}_info ADD COLUMN 楼层类型 TEXT;" , 
               f"ALTER TABLE {key}_info ADD COLUMN 楼层数 INTEGER;",
               f"UPDATE {key}_info" +  
                """SET 
                    楼层类型 = SUBSTR(所在楼层, 1, 
                                    INSTR(所在楼层, '（') - 1),  
                    楼层数 = CAST(SUBSTR(所在楼层, INSTR(所在楼层, '共') + 1, 
                                INSTR(所在楼层, '层') - 1) AS INTEGER);"""]

        for s in sql:
            with sqlite3.connect(db.db_path) as conn:
                conn.execute(s)

    sql = """SELECT 梯户比例, count(*) cnt
            FROM beicai_info
            GROUP BY 梯户比例
            ORDER BY cnt DESC
            LIMIT 25;"""
    print(db.show_data(sql=sql))

    sql = """SELECT 房屋年限, count(*) cnt
            FROM beicai_info
            GROUP BY 房屋年限
            ORDER BY cnt DESC
            LIMIT 25;"""
    print(db.show_data(sql=sql))

    sql = """SELECT AVG(建成年代)
            FROM beicai_info;"""
    print(db.show_data(sql=sql))

