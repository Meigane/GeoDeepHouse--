import streamlit as st 
import sys
from pathlib import Path  
  
module_path = Path(__file__).resolve().parent.parent / 'app' / 'pages'  
sys.path.append(str(module_path))   

st.title("🚀 GeoDeepHouse - 基于时空大数据的智能房价预测系统 🚀")  

# 展示数据处理与分析的pipeline  
st.markdown("""  
### 🔧 数据处理与分析Pipeline 🔧  

1. **数据采集** 🌐  
- 使用Selenium从链家贝壳网抓取二手房数据,并储存在sqllite数据库中。  
- 收集POI数据，包括地铁、医院、学校等。  
- 获取GIS数据和表格数据。  

2. **数据处理** 🧹  
- 利用sql清洗和整理数据。
- 对文本数据进行标签编码  
- 使用GeoPandas处理地理数据。
- 使用百度地图api和阿里云平台进行地理编码  

3. **数据分析** 📊  
- 进行空间热点分析、价格趋势分析和POI数据分析。
- 构建了深度习模型进行房价预测。  
- 尝试进行了核密度分析以及IDW插值，但由于时间复杂度较高，并未在Web应用中应用。  

4. **数据可视化** 📈  
- 使用Matplotlib、Seaborn和Folium进行数据可视化。  

5. **应用开发** 💻  
- 基于Streamlit开发交互式Web应用。

""")  

# 展示应用功能页面概述  
st.markdown("""  
### 🌐 应用功能页面概览 🌐  

- **空间热点分析** 🔥  
- 全市房价热力图展示。  
- 各区成交房数量展示。
- 可通过地图控件交互  

- **成交房数据探索性分析** 📍  
- 时间序列分析。  
- 相关性分析。  
- 柱状图散点图分析。

- **了解行情** 📊  
- 交互页面，用户选择区域查看成交房价分布和趋势。

- **看看周边** 📊  
- 用户可以查看心意小区附近的医疗、教育、交通设施  

- **房价预测** 🏠 
- 用户上传房屋特征，基于深度学习模型预测房价。  
- 提供房源周边设施信息和交互式地图(开发中敬请期待。。。。) 。  
""")  