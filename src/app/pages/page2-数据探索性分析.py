import streamlit as st  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib  
import seaborn as sns  
import sqlite3  
import geopandas as gpd  
from pathlib import Path 
# import plotly.express as px   
  
# 设置路径  
db_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed")  
gis_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "gis_data")  
  
# 连接SQLite数据库  
conn = sqlite3.connect(db_path + '//GisProject.db')  
  
# 读取数据  
df1 = pd.read_sql_query("SELECT * FROM house_info", conn)  
df2 = pd.read_sql_query("SELECT * FROM communities", conn)  
  
# 读取shapefile  
gdf = gpd.read_file(gis_path + '//communities.shp')  
  
# 合并数据库和shapefile中的数据  
df1 = df1.merge(gdf[['name', 'cnt', '区']], left_on='小区', right_on='name', how='left')  
  
# 设置支持中文的字体  
matplotlib.rcParams['font.family'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号  
 
st.title("探索性分析") 
# 时间序列分析：挂牌时间趋势分析  
st.subheader("挂牌价格时间趋势分析")  
df1['挂牌时间'] = pd.to_datetime(df1['挂牌时间'])  
df1.set_index('挂牌时间', inplace=True)  
# 计算每月的平均挂牌价格  
monthly_prices = df1.resample('M')['单价'].mean()  
st.line_chart(monthly_prices)
  
# 挂牌时间与挂牌数量的深度分析  
st.subheader("挂牌时间与挂牌数量的关系")  
df1['挂牌年份'] = df1.index.year  
annual_listings = df1.groupby('挂牌年份').size()  
st.bar_chart(annual_listings) 

# 相关性分析部分 
fig, axs = plt.subplots(1, 2, figsize=(14, 7))  
df1 = df1.rename(columns={'cnt': '售出总数'})  
corr_matrix = df1[['总价', '单价', '面积', '室数', '厅数', '楼层数', '建成年代', '售出总数']].corr()  
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axs[0])  
  
# 柱状图：展示总价与室数的关系  
st.subheader("相关性分析")  
sns.barplot(x='室数', y='总价', data=df1.sort_values('室数'), ax=axs[1])  
st.pyplot(fig)  
  
# 散点图：展示面积与单价的关系，并区分不同的区  
st.subheader("面积与单价的关系（区分不同区）")  
fig, axs = plt.subplots()  
sns.scatterplot(x='面积', y='单价', hue='区', data=df1, ax=axs)  
axs.set_xlim(0, 500)  
axs.set_ylim(0, 200000)  
st.pyplot(fig) 

# st.subheader("面积与单价的关系（区分不同区）")  
  
# # 利用plotly.express创建可交互的散点图  
# fig = px.scatter(df1, x='面积', y='单价', color='区', size_max=20)  
  
# # 设置x轴和y轴的显示范围  
# fig.update_layout(  
#     xaxis=dict(range=[0, 500]),  # 设置x轴的范围为0到500  
#     yaxis=dict(range=[0, 200000])  # 设置y轴的范围为0到200000  
# )  
  
# # 使用streamlit的plotly_chart函数来直接展示plotly图表，以提供完整的交互功能  
# st.plotly_chart(fig)


