import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_house_data():
    """加载房屋数据"""
    try:
        base_path = Path(__file__).resolve().parent.parent.parent
        db_path = base_path / "data/processed/GisProject.db"
        conn = sqlite3.connect(db_path)
        house_df = pd.read_sql("SELECT * FROM house_info", conn)
        print(f"成功加载 {len(house_df)} 条房屋数据")
        return house_df, conn
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def calculate_area_prices(house_df, conn):
    """计算并存储区域平均房价"""
    try:
        # 计算区域平均房价
        area_prices = house_df.groupby('所在地区')['单价'].agg(['mean', 'count']).reset_index()
        area_prices.columns = ['区域', '平均房价', '房屋数量']
        
        # 创建区域房价表
        print("创建区域房价表...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS area_prices (
                区域 TEXT PRIMARY KEY,
                平均房价 REAL,
                房屋数量 INTEGER,
                更新时间 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 保存到数据库
        area_prices['更新时间'] = pd.Timestamp.now()
        area_prices.to_sql('area_prices', conn, if_exists='replace', index=False)
        
        print("区域房价统计信息：")
        print(area_prices)
        return area_prices
        
    except Exception as e:
        print(f"计算区域平均房价时出错: {str(e)}")
        return None

def process_batch(batch_df, area_prices):
    """处理一批数据"""
    try:
        # 使用预计算的区域平均房价
        batch_df = batch_df.merge(
            area_prices[['区域', '平均房价']], 
            left_on='所在地区', 
            right_on='区域', 
            how='left'
        )
        batch_df['区域平均房价'] = batch_df['平均房价']
        batch_df = batch_df.drop(['区域', '平均房价'], axis=1)
        
        # 计算建筑年龄
        current_year = datetime.now().year
        batch_df['建筑年龄'] = current_year - batch_df['建成年代']
        
        # 提取季节信息
        batch_df['季节性特征'] = pd.to_datetime(batch_df['挂牌时间']).dt.quarter
        
        # 房龄分段编码
        def age_category(age):
            if age <= 5:
                return 0  # 新房
            elif age <= 15:
                return 1  # 准新房
            elif age <= 30:
                return 2  # 普通住宅
            else:
                return 3  # 老房
        
        batch_df['房龄分段'] = batch_df['建筑年龄'].apply(age_category)
        
        # 户型面积比
        batch_df['户型面积比'] = batch_df['面积'] / batch_df['室数'].replace(0, np.nan)
        
        # 每室价格
        batch_df['每室价格'] = batch_df['总价'] / batch_df['室数'].replace(0, np.nan)
        
        # 楼层分类编码
        def floor_category(floor):
            try:
                floor = int(floor)
                if floor <= 6:
                    return 0  # 低楼层
                elif floor <= 18:
                    return 1  # 中楼层
                else:
                    return 2  # 高楼层
            except:
                return -1  # 未知
        
        batch_df['楼层分类'] = batch_df['楼层数'].apply(floor_category)
        
        # 装修情况编码
        decoration_map = {
            '其他': 0,
            '简装': 1,
            '精装': 2
        }
        batch_df['装修情况编码'] = batch_df['装修情况'].map(decoration_map)
        
        # 朝向评分
        direction_scores = {
            '南': 1.0,    # 最佳朝向
            '东南': 0.9,  # 次佳朝向
            '东': 0.8,
            '西南': 0.7,
            '东北': 0.6,
            '西': 0.5,
            '北': 0.4,
            '西北': 0.3,
            '无': 0.0     # 未知朝向
        }
        batch_df['朝向评分'] = batch_df['房屋朝向'].map(direction_scores)
        
        return batch_df
    
    except Exception as e:
        print(f"处理批次数据时出错: {str(e)}")
        return None

def update_database(df, conn):
    """更新数据库"""
    try:
        # 获取需要添加的新列
        new_columns = [
            '区域平均房价', '建筑年龄', '季节性特征', '房龄分段',
            '户型面积比', '每室价格', '楼层分类', '装修情况编码', '朝向评分'
        ]
        
        # 获取现有表结构
        existing_columns = pd.read_sql("PRAGMA table_info(house_info);", conn)['name'].tolist()
        
        print("开始更新数据库...")
        for column in tqdm(new_columns, desc="更新列"):
            try:
                # 如果列不存在，添加列
                if column not in existing_columns:
                    print(f"\n添加新列: {column}")
                    conn.execute(f"ALTER TABLE house_info ADD COLUMN {column} REAL;")
                
                # 分批更新数据
                batch_size = 1000
                total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
                
                print(f"\n开始更新列 {column} 的数据...")
                batch_progress = tqdm(total=total_batches, desc=f"更新 {column}", position=1, leave=False)
                
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    
                    # 使用参数化查询避免SQL注入
                    update_sql = f"""
                        UPDATE house_info 
                        SET {column} = ? 
                        WHERE 链家编号 = ?;
                    """
                    
                    # 准备批量更新数据
                    update_data = [
                        (row[column], row['链家编号']) 
                        for _, row in batch_df.iterrows()
                    ]
                    
                    # 执行批量更新
                    conn.executemany(update_sql, update_data)
                    
                    # 每1000条提交一次
                    if i % (batch_size * 10) == 0:
                        conn.commit()
                        print(f"\n已更新 {i+len(batch_df)}/{len(df)} 条记录")
                    
                    batch_progress.update(1)
                
                conn.commit()
                batch_progress.close()
                print(f"\n列 {column} 更新完成")
                
            except Exception as e:
                print(f"\n更新列 {column} 时出错: {str(e)}")
                continue
        
    except Exception as e:
        print(f"\n更新数据库时出错: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        print("开始加载数据...")
        house_df, conn = load_house_data()
        
        # 首先计算区域平均房价
        print("\n计算区域平均房价...")
        area_prices = calculate_area_prices(house_df, conn)
        if area_prices is None:
            raise Exception("区域平均房价计算失败")
        
        print("\n开始数据处理...")
        # 分批处理数据
        BATCH_SIZE = 5000
        processed_dfs = []
        total_batches = len(house_df) // BATCH_SIZE + (1 if len(house_df) % BATCH_SIZE else 0)
        
        batch_progress = tqdm(total=total_batches, desc="处理数据批次")
        
        for i in range(0, len(house_df), BATCH_SIZE):
            print(f"\n处理第 {i//BATCH_SIZE + 1}/{total_batches} 批数据")
            batch = house_df.iloc[i:i+BATCH_SIZE].copy()
            processed_batch = process_batch(batch, area_prices)  # 传入预计算的区域平均房价
            if processed_batch is not None:
                processed_dfs.append(processed_batch)
            batch_progress.update(1)
        
        batch_progress.close()
        
        print("\n合并处理后的数据...")
        processed_df = pd.concat(processed_dfs, ignore_index=True)
        
        # 更新数据库
        update_database(processed_df, conn)
        
        # 显示特征统计信息
        print("\n特征统计信息:")
        print(processed_df[[
            '建筑年龄', '户型面积比', 
            '每室价格', '朝向评分', '房龄分段', 
            '楼层分类', '装修情况编码'
        ]].describe())
        
        conn.close()
        print("\n特征计算和保存完成!")
        
    except Exception as e:
        print(f"\n特征提取过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

if __name__ == "__main__":
    main()
