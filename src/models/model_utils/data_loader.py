import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class HousePriceDataset(Dataset):
    """PyTorch数据集类"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class HousePriceDataLoader:
    def __init__(self, db_path=None):
        if db_path is None:
            base_path = Path(__file__).resolve().parent.parent.parent.parent
            db_path = base_path / "data/processed/GisProject.db"
        self.db_path = db_path
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def load_data(self):
        """加载并预处理数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 加载房屋信息和小区信息
        house_info = pd.read_sql("""
            SELECT h.*, 
                   c.距离市中心, c.距离最近地铁, c.地铁站数量,
                   c.教育资源密度, c.医疗设施密度, c.商业设施密度
            FROM house_info h
            LEFT JOIN communities c ON h.小区 = c.name
        """, conn)
        
        conn.close()
        
        # 选择用于训练的特征
        selected_features = [
            # 数值特征
            '面积', '建成年代', '楼层数', '室数', '厅数',
            # 工程特征
            '区域平均房价', '建筑年龄', '季节性特征', '房龄分段',
            '户型面积比', '每室价格', '楼层分类', '装修情况编码', '朝向评分',
            # 位置特征
            '距离市中心', '距离最近地铁', '地铁站数量',
            '教育资源密度', '医疗设施密度', '商业设施密度'
        ]
        
        # 目标变量
        target = '单价'
        
        # 去除包含空值的行
        data = house_info[selected_features + [target]].dropna()
        
        # 修改异常值处理逻辑
        def remove_price_outliers(df, price_column='单价'):
            """去除房价异常值
            
            上海二手房单价参考标准（2024）：
            - 最低：约15000元/平米（远郊区域）
            - 最高：约120000元/平米（市中心豪宅）
            - 常见区间：30000-80000元/平米
            """
            # 设定合理的价格区间
            MIN_PRICE = 15000  # 最低单价（元/平米）
            MAX_PRICE = 120000  # 最高单价（元/平米）
            
            # 打印异常值处理前的统计信息
            print("\n异常值处理前的房价统计:")
            print(f"数据量: {len(df)}")
            print(f"最小值: {df[price_column].min():.2f}")
            print(f"最大值: {df[price_column].max():.2f}")
            print(f"均值: {df[price_column].mean():.2f}")
            print(f"中位数: {df[price_column].median():.2f}")
            
            # 使用价格区间过滤
            df_filtered = df[
                (df[price_column] >= MIN_PRICE) & 
                (df[price_column] <= MAX_PRICE)
            ]
            
            # 进一步使用IQR方法去除剩余的异常值
            Q1 = df_filtered[price_column].quantile(0.25)
            Q3 = df_filtered[price_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 确保边界值在合理范围内
            lower_bound = max(lower_bound, MIN_PRICE)
            upper_bound = min(upper_bound, MAX_PRICE)
            
            df_filtered = df_filtered[
                (df_filtered[price_column] >= lower_bound) & 
                (df_filtered[price_column] <= upper_bound)
            ]
            
            # 打印异常值处理后的统计信息
            print("\n异常值处理后的房价统计:")
            print(f"数据量: {len(df_filtered)}")
            print(f"最小值: {df_filtered[price_column].min():.2f}")
            print(f"最大值: {df_filtered[price_column].max():.2f}")
            print(f"均值: {df_filtered[price_column].mean():.2f}")
            print(f"中位数: {df_filtered[price_column].median():.2f}")
            print(f"移除的数据比例: {(len(df) - len(df_filtered)) / len(df) * 100:.2f}%")
            
            return df_filtered
        
        # 使用新的异常值处理函数
        data = remove_price_outliers(data)
        
        # 对其他数值特征进行异常值处理
        numerical_features = [
            '面积', '建成年代', '楼层数', '距离市中心', 
            '距离最近地铁', '教育资源密度', '医疗设施密度', '商业设施密度'
        ]
        
        def remove_feature_outliers(df, column, n_std=3):
            """使用z-score方法处理其他特征的异常值"""
            mean = df[column].mean()
            std = df[column].std()
            df_filtered = df[abs(df[column] - mean) <= (n_std * std)]
            removed_ratio = (len(df) - len(df_filtered)) / len(df) * 100
            print(f"\n{column} 异常值处理:")
            print(f"移除的数据比例: {removed_ratio:.2f}%")
            return df_filtered
        
        # 对每个数值特征进行异常值处理
        for feature in numerical_features:
            data = remove_feature_outliers(data, feature)
        
        # 更新特征和目标变量
        X = data[selected_features]
        y = data[target]
        
        print("\n原始房价统计:")
        print(f"最小值: {y.min():.2f}")
        print(f"最大值: {y.max():.2f}")
        print(f"均值: {y.mean():.2f}")
        print(f"标准差: {y.std():.2f}")
        
        # 特征归一化前打印统计信息
        print("\n原始特征统计:")
        for col in X.columns:
            print(f"{col}:")
            print(f"  最小值: {X[col].min():.2f}")
            print(f"  最大值: {X[col].max():.2f}")
            print(f"  均值: {X[col].mean():.2f}")
        
        # 特征归一化
        X_scaled = self.scaler_x.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 打印特征维度信息
        print("\n特征维度信息:")
        print(f"特征数量: {X_scaled_df.shape[1]}")
        print(f"样本数量: {X_scaled_df.shape[0]}")
        print("特征列表:")
        for i, col in enumerate(X_scaled_df.columns):
            print(f"{i+1}. {col}")
        
        # 确保所有特征值都在合理范围内
        assert not np.any(np.isnan(X_scaled_df.values)), "数据中包含NaN值"
        assert not np.any(np.isinf(X_scaled_df.values)), "数据中包含无穷大值"
        
        # 检查是否所有特征都已经归一化到[0,1]范围
        min_vals = X_scaled_df.min()
        max_vals = X_scaled_df.max()
        if not (min_vals.between(-1e-6, 1+1e-6).all() and max_vals.between(-1e-6, 1+1e-6).all()):
            print("\n警告：某些特征的归一化范围异常:")
            for col in X_scaled_df.columns:
                if not (X_scaled_df[col].between(-1e-6, 1+1e-6).all()):
                    print(f"{col}: min={X_scaled_df[col].min():.4f}, max={X_scaled_df[col].max():.4f}")
        
        # 目标变量处理
        y_log = np.log1p(y)
        y_scaled = self.scaler_y.fit_transform(y_log.values.reshape(-1, 1)).ravel()
        
        # 检查目标变量
        print("\n目标变量维度信息:")
        print(f"样本数量: {len(y_scaled)}")
        assert not np.any(np.isnan(y_scaled)), "目标变量中包含NaN值"
        assert not np.any(np.isinf(y_scaled)), "目标变量中包含无穷大值"
        
        return X_scaled_df, y_scaled
    
    def get_data_loaders(self, batch_size=32, train_split=0.8, val_split=0.1):
        """获取训练、验证和测试数据加载器"""
        X, y = self.load_data()
        
        # 确保batch_size合理
        total_samples = len(X)
        if batch_size > total_samples // 10:
            print(f"\n警告: batch_size ({batch_size}) 可能过大，自动调整")
            batch_size = max(1, total_samples // 32)
            print(f"调整后的batch_size: {batch_size}")
        
        # 打印数据集维度信息
        print("\n数据集维度信息:")
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")
        
        # 印数据集大小信息
        print("\n数据集划分信息:")
        print(f"总数据量: {len(X)}")
        print(f"Batch大小: {batch_size}")
        print(f"预计batch数量: {len(X) // batch_size}")
        
        # 确保X和y是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # 计算数据集大小
        total_size = len(X)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size
        
        print(f"\n训练集大小: {train_size} ({train_split*100}%)")
        print(f"验证集大小: {val_size} ({val_split*100}%)")
        print(f"测试集大小: {test_size} ({(1-train_split-val_split)*100}%)")
        
        # 创建数据集索引
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # 创建数据集
        full_dataset = HousePriceDataset(X, y)  # 直接使用numpy数组
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print("\n数据加载器信息:")
        print(f"训练batch数: {len(train_loader)}")
        print(f"验证batch数: {len(val_loader)}")
        print(f"测试batch数: {len(test_loader)}")
        
        # 创建数据加载器时添加额外的检查
        for i, (features, labels) in enumerate(train_loader):
            print(f"\n检查第一个batch的维度:")
            print(f"特征维度: {features.shape}")
            print(f"标签维度: {labels.shape}")
            break
        
        return train_loader, val_loader, test_loader
    
    def get_feature_names(self):
        """获取特征名称"""
        X, _ = self.load_data()
        return X.columns.tolist()
    
    def inverse_transform_predictions(self, scaled_predictions):
        """将归一化的预测值转换回原始范围"""
        try:
            print("\n反归一化过程:")
            print("1. 输入的归一化预测值:", scaled_predictions)
            
            # 先反归一化到对数空间
            log_predictions = self.scaler_y.inverse_transform(scaled_predictions.reshape(-1, 1))
            print("2. 反归一化到对数空间:", log_predictions.ravel())
            
            # 再反对数转换到原始空间
            original_predictions = np.expm1(log_predictions).ravel()
            print("3. 反对数转换到原始空间:", original_predictions)
            
            # 打印scaler的参数
            print("\nScaler参数:")
            print("min_:", self.scaler_y.min_)
            print("scale_:", self.scaler_y.scale_)
            
            return original_predictions
        except Exception as e:
            print(f"反归一化出错: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            return scaled_predictions

# 使用示例
if __name__ == "__main__":
    # 创建数据加载器
    loader = HousePriceDataLoader()
    
    # 获取处理后的数据
    X, y = loader.load_data()
    print("\n数据形状:")
    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    
    # 显示特征统计信息
    print("\n特征统计信息:")
    print(X.describe())
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = loader.get_data_loaders(batch_size=32)
    print("\n数据加载器信息:")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 检查一个批次的数据
    for features, labels in train_loader:
        print("\n批次数据形状:")
        print(f"特征形状: {features.shape}")
        print(f"标签形状: {labels.shape}")
        break
