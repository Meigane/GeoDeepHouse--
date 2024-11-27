import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import sys

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# 使用相对导入
from src.models.model_utils.data_loader import HousePriceDataLoader
from src.models.deep_learning.dcn_model import create_dcn_model

def load_model():
    """加载训练好的模型"""
    try:
        model_path = Path(r"F:\files\code\DL\best_dcn_model.pth")
        
        print("文件是否存在:", model_path.exists())
        print("文件大小:", model_path.stat().st_size if model_path.exists() else "文件不存在")
        
        # 检查文件是否存在
        if not model_path.exists():
            st.error(f"模型文件不存在: {model_path}")
            return None, None
            
        # 创建数据加载器以获取特征数量
        loader = HousePriceDataLoader()
        input_dim = len(loader.get_feature_names())
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_dcn_model(input_dim, device)
        
        try:
            # 加载模型参数
            checkpoint = torch.load(str(model_path), map_location=device)
            
            # 打印检查点内容
            print("\n检查点内容:")
            for key in checkpoint:
                print(f"Key: {key}")
            
            # 加载模型状态字典
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 打印模型参数统计
            print("\n模型参数统计:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}:")
                    print(f"  形状: {param.shape}")
                    print(f"  均值: {param.mean().item():.4f}")
                    print(f"  标准差: {param.std().item():.4f}")
            
            model.eval()
            st.success("模型加载成功!")
            return model, loader
            
        except Exception as e:
            st.error(f"加载模型参数时出错: {str(e)}")
            import traceback
            st.error(f"详细错误信息:\n{traceback.format_exc()}")
            return None, None
            
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        import traceback
        st.error(f"详细错误信息:\n{traceback.format_exc()}")
        return None, None

def predict_price(model, features, loader):
    """使用PyTorch模型进行预测"""
    try:
        with torch.no_grad():
            # 打印原始特征
            print("\n预测过程:")
            print("1. 原始特征:", features)
            
            # 使用训练时的scaler进行特征归一化
            features_normalized = loader.scaler_x.transform(features.reshape(1, -1))
            print("2. 归一化后的特征:", features_normalized)
            
            # 转换为tensor
            features_tensor = torch.FloatTensor(features_normalized)
            print("3. Tensor形状:", features_tensor.shape)
            
            # 检查模型状态
            print("\n模型状态:")
            print("模型设备:", next(model.parameters()).device)
            print("输入设备:", features_tensor.device)
            
            # 预测
            model.eval()
            output = model(features_tensor)
            print("4. 模型原始输出:", output.numpy())
            
            # 反归一化
            price = loader.inverse_transform_predictions(output.numpy())[0]
            print("5. 最终预测价格:", price)
            
            return price
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return None

def get_districts():
    """从数据库获取所有地区"""
    try:
        db_path = r"F:\课件\第四学期\GIS开发\大作业\gis_project_linglingzhang\data\processed\GisProject.db"
        conn = sqlite3.connect(db_path)
        districts = pd.read_sql("""
            SELECT DISTINCT 所在地区
            FROM house_info
            WHERE 所在地区 IS NOT NULL
            ORDER BY 所在地区
        """, conn)['所在地区'].tolist()
        conn.close()
        return districts
    except Exception as e:
        st.error(f"获取地区列表时出错: {str(e)}")
        return ["浦东新区"]  # 返回默认值

def create_prediction_page():
    st.title("房价预测系统 🏠")
    
    # 加载模型
    model, loader = load_model()
    if model is None or loader is None:
        st.error("模型加载失败")
        return
    
    # 获取地区列表
    districts = get_districts()
    
    # 创建输入表单
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input("面积 (平方米)", min_value=10.0, max_value=500.0, value=90.0)
            district = st.selectbox("所在地区", districts)  # 使用从数据库获取的地区列表
            rooms = st.number_input("室数", min_value=1, max_value=6, value=2)
            halls = st.number_input("厅数", min_value=0, max_value=3, value=1)
            
        with col2:
            floor = st.number_input("楼层数", min_value=1, max_value=50, value=6)
            building_age = st.number_input("建筑年代", min_value=1980, max_value=2024, value=2000)
            decoration = st.selectbox("装修情况", ["精装", "简装", "其他"])
            direction = st.selectbox("房屋朝向", ["南", "东南", "东", "西南", "东北", "西", "北", "西北"])
        
        submitted = st.form_submit_button("预测房价")
    
    if submitted:
        try:
            # 准备所有需要的特征
            current_year = 2024
            
            # 从数据库获取区域平均房价
            db_path = r"F:\课件\第四学期\GIS开发\大作业\gis_project_linglingzhang\data\processed\GisProject.db"
            conn = sqlite3.connect(db_path)
            avg_price = pd.read_sql(f"""
                SELECT AVG(单价) as avg_price
                FROM house_info
                WHERE 所在地区 = '{district}'
            """, conn).iloc[0]['avg_price']
            conn.close()
            
            # 构建特征字典
            features = {
                '面积': area,
                '建成年代': building_age,
                '楼层数': floor,
                '室数': rooms,
                '厅数': halls,
                '区域平均房价': avg_price,
                '建筑年龄': current_year - building_age,
                '季节性特征': pd.Timestamp.now().quarter,  # 使用当前季度
                '房龄分段': 0 if current_year - building_age <= 5 else (
                    1 if current_year - building_age <= 15 else (
                    2 if current_year - building_age <= 30 else 3
                )),
                '户型面积比': area / rooms,
                '每室价格': avg_price * area / rooms,  # 使用区域平均价
                '楼层分类': 0 if floor <= 6 else (1 if floor <= 18 else 2),
                '装修情况编码': {'其他': 0, '简装': 1, '精装': 2}[decoration],
                '朝向评分': {'南': 1.0, '东南': 0.9, '东': 0.8, '西南': 0.7,
                         '东北': 0.6, '西': 0.5, '北': 0.4, '西北': 0.3}[direction],
            }
            
            # 从数据库获取位置特征
            conn = sqlite3.connect(db_path)
            location_features = pd.read_sql(f"""
                SELECT 距离市中心, 距离最近地铁, 地铁站数量,
                       教育资源密度, 医疗设施密度, 商业设施密度
                FROM communities
                WHERE name IN (
                    SELECT 小区
                    FROM house_info
                    WHERE 所在地区 = '{district}'
                    LIMIT 1
                )
            """, conn).iloc[0].to_dict()
            conn.close()
            
            # 更新位置特征
            features.update(location_features)
            
            # 确保特征顺序与训练时一致
            feature_names = loader.get_feature_names()
            st.write("特征名称:", feature_names)
            
            feature_values = [features[name] for name in feature_names]
            st.write("特征值:", feature_values)
            
            # 检查特征范围
            st.write("特征统计:")
            st.write("最小值:", np.min(feature_values))
            st.write("最大值:", np.max(feature_values))
            st.write("均值:", np.mean(feature_values))
            
            # 预测
            pred_price = predict_price(model, np.array(feature_values), loader)
            
            if pred_price is not None:
                # 显示预测结果
                st.success(f"预测单价: {pred_price:,.2f} 元/平方米")
                total_price = pred_price * area / 10000
                st.success(f"预测总价: {total_price:,.2f} 万元")
                
                # 添加置信区间
                confidence = 0.95
                std_dev = pred_price * 0.1  # 假设标准差为预测值的10%
                z_score = 1.96  # 95%置信区间的z分数
                lower_bound = pred_price - z_score * std_dev
                upper_bound = pred_price + z_score * std_dev
                
                st.write("预测区间:")
                st.write(f"下限: {lower_bound:,.2f} 元/平方米")
                st.write(f"上限: {upper_bound:,.2f} 元/平方米")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
            st.error("特征名称:", feature_names)
            st.error("特征值:", feature_values)

if __name__ == "__main__":
    create_prediction_page() 