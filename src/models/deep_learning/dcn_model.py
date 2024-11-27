import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

class CrossLayer(nn.Module):
    """改进的交叉层实现"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # 使用正态分布初始化权重，标准差较小以防止过拟合
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        # 添加Layer Normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x0, x):
        xw = torch.matmul(x, self.weight)
        xw = xw.unsqueeze(1)
        cross_term = x0 * xw
        # 添加Layer Normalization
        output = self.layer_norm(cross_term + self.bias + x)
        return output

class ResidualBlock(nn.Module):
    """改进的残差块"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 增加中间层维度
            nn.LayerNorm(dim * 4),    # 使用Layer Normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim)         # 使用Layer Normalization
        )
        # 添加门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 门控残差连接
        out = self.layers(x)
        gate_value = self.gate(x)
        return x + gate_value * out

class DCN(nn.Module):
    def __init__(self, input_dim, num_cross_layers=4, deep_layers=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        # 添加Batch Normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # 改进特征嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        
        # 初始化交叉层
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])
        
        # 深度网络层
        self.deep_layers = nn.ModuleList()
        current_dim = input_dim
        for dim in deep_layers:
            self.deep_layers.append(nn.Sequential(
                nn.Linear(current_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            current_dim = dim
        
        # 添加维度转换层，确保注意力机制的输入维度正确
        self.attention_projection = nn.Linear(deep_layers[-1], 64)
        
        # 修改注意力机制的维度
        self.attention = nn.MultiheadAttention(
            embed_dim=64,  # 改为64维
            num_heads=4,
            dropout=dropout
        )
        
        # 修改输出层的输入维度
        self.combine_layer = nn.Sequential(
            nn.Linear(input_dim + 64, 128),  # 修改为64
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.final_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
        # 添加L2正则化
        self.l2_reg = 0.01
        
        # 修改损失权重
        self.loss_weights = {
            'mse': 0.2,
            'mae': 0.2,
            'r2': 0.5,
            'l2': 0.1
        }
    
    def r2_loss(self, pred, target):
        """计算R²损失"""
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 计算R²
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        
        r2 = 1 - ss_res / (ss_tot + 1e-8)  # 添加小值防止除零
        
        # 转换为损失（R²越大越好，所以用1减去R²）
        return 1 - r2
    
    def custom_loss(self, pred, target):
        """改进的组合损失函数"""
        # 确保维度匹配
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 忽略警告
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using a target size.*")
            
            # MSE损失
            mse_loss = F.mse_loss(pred, target)
            
            # MAE损失
            mae_loss = F.l1_loss(pred, target)
            
            # R²损失
            r2_loss = self.r2_loss(pred, target)
            
            # L2正则化损失
            l2_loss = self.get_l2_loss()
        
        # 组合损失
        total_loss = (
            self.loss_weights['mse'] * mse_loss +
            self.loss_weights['mae'] * mae_loss +
            self.loss_weights['r2'] * r2_loss +
            self.loss_weights['l2'] * l2_loss
        )
        
        # 打印各个损失分量（用于调试）
        if torch.rand(1).item() < 0.001:  # 1%的概率打印
            print(f"\nLoss components:")
            print(f"MSE: {mse_loss.item():.4f}")
            print(f"MAE: {mae_loss.item():.4f}")
            print(f"R²: {(1 - r2_loss.item()):.4f}")
            print(f"L2: {l2_loss.item():.4f}")
        
        return total_loss
        
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_l2_loss(self):
        """计算L2正则化损失"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss

    def forward(self, x):
        # 批量归一化
        x = self.batch_norm(x)
        
        # 特征嵌入
        x = self.embedding(x)
        x0 = x
        
        # 交叉网络
        cross_output = x
        for cross_layer in self.cross_layers:
            cross_output = cross_layer(x0, cross_output)
        
        # 深度网络
        deep_output = x
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
        
        # 投影到64维空间
        deep_output = self.attention_projection(deep_output)  # [batch_size, 64]
        
        # 注意力机制
        deep_output = deep_output.unsqueeze(0)  # [1, batch_size, 64]
        deep_output, _ = self.attention(deep_output, deep_output, deep_output)
        deep_output = deep_output.squeeze(0)  # [batch_size, 64]
        
        # 合并交叉网络和深度网络的输出
        combined = torch.cat([cross_output, deep_output], dim=1)
        
        # 最终预测
        combined = self.combine_layer(combined)
        output = self.final_layer(combined)
        
        return output  # 输出形状为 [batch_size, 1]

def create_dcn_model(input_dim, device='cuda'):
    """创建改进的DCN模型"""
    model = DCN(
        input_dim=input_dim,
        num_cross_layers=4,
        deep_layers=[256, 128, 64],
        dropout=0.3
    )
    return model.to(device)

# 测试代码
if __name__ == "__main__":
    # 使用实际特征数量
    input_dim = 20  # 与data_loader中的特征数量相匹配
    batch_size = 32
    
    # 创建模型
    model = create_dcn_model(input_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成测试数据
    x = torch.randn(batch_size, input_dim)
    if torch.cuda.is_available():
        x = x.cuda()
    
    # 前向传播
    output = model(x)
    
    # print("\n模型结构:")
    # print(model)
    # print("\n输入形状:", x.shape)
    # print("输出形状:", output.shape)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,}") 