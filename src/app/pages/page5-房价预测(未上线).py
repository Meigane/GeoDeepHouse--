import streamlit as st  
import pandas as pd  
from sklearn.preprocessing import LabelEncoder 
from pathlib import Path  
import sys
  
module_path = Path(__file__).resolve().parent.parent / 'data_analysis'  
file_path = Path(__file__).resolve().parent.parent.parent / "models"
sys.path.append(str(module_path))  
from HousePricePredictor import HousePricePredictor    
import joblib  
import os  
  
model_name = 'decision_tree_regressor'  
encoder_file = 'encoders.joblib'  # LabelEncoder映射保存的文件名  
  
# 加载模型和LabelEncoder映射  
model_path = os.path.join(file_path, model_name + '.joblib')  
encoder_path = os.path.join(file_path, encoder_file)  
model = joblib.load(model_path)  
encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}  
  
# Streamlit 应用  
st.title("房价预测应用")  
  
# 用户输入  
house_info = {}  
all_columns = ['房型', '面积', '房屋户型', '所在楼层', '户型结构', '建筑类型', '房屋朝向', '建成年代', '装修情况', '建筑结构', '梯户比例', '配备电梯', '交易权属', '房屋用途', '房权所属', '楼层数', '所在地区']  
  
for col in all_columns:  
    if col == '面积':  
        house_info[col] = st.number_input(f"{col}（平方米）")  
    elif col in ['所在楼层', '建成年代', '楼层数']:  
        house_info[col] = st.number_input(f"{col}（整数）")  
    else:  

        options = {'房型': ['三室一厅', '两室一厅', '一室一厅'],   
                   '房屋户型': ['南北通透', '朝南', '朝北'],  
                   '所在楼层': [],  # 这不是文本列，但保留在循环中以简化代码  
                   # ... 其他文本列的选项  
                   }  
        house_info[col] = st.selectbox(f"{col}", options.get(col, []))  
  
# 将用户输入映射到数值（仅对文本列）  
for col in [col for col in all_columns if col in encoders]:  
    if col in house_info:  
        house_info[col] = encoders[col].transform([house_info[col]])[0]  
  
# 创建DataFrame行  
input_row = pd.Series(house_info, index=all_columns)  
input_df = pd.DataFrame([input_row])  
  
# 预测  
prediction = model.predict(input_df)[0]  
  
# 显示预测结果  
st.write(f"预测的总价为：{prediction:.2f}万元")  
  