import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np
import sys
import pickle
import joblib
from torch.utils.data import DataLoader
from train_utils import EarlyStopping, calculate_metrics, AdaptiveLRScheduler
import copy

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parent.parent.parent
# project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.models.model_utils.data_loader import HousePriceDataLoader
from src.models.deep_learning.dcn_model import create_dcn_model

class RelativeL1Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + self.epsilon))

class ModelTrainer:
    def __init__(self, model, device='cuda', save_dir=None):
        self.model = model
        self.device = device
        if save_dir is None:
            # 使用绝对路径，并确保创建完整的目录结构
            project_root = Path(__file__).resolve().parent.parent.parent
            save_dir = project_root / "models" / "saved_models"
        self.save_dir = Path(save_dir)
        # 创建完整的目录结构
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化最佳指标
        self.best_val_loss = float('inf')
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, early_stopping=10):
        """训练模型"""
        criterion = RelativeL1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        # 早停计数器
        no_improve = 0
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_mae = 0
            train_batches = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for features, targets in train_pbar:
                features, targets = features.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()
                train_batches += 1
                
                train_pbar.set_postfix({
                    'loss': train_loss/train_batches,
                    'mae': train_mae/train_batches
                })
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_mae = 0
            val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for features, targets in val_pbar:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features).squeeze()
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - targets)).item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({
                        'loss': val_loss/val_batches,
                        'mae': val_mae/val_batches
                    })
            
            # 计算平均损失
            avg_train_loss = train_loss / train_batches
            avg_train_mae = train_mae / train_batches
            avg_val_loss = val_loss / val_batches
            avg_val_mae = val_mae / val_batches
            
            # 更新历史记录
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mae'].append(avg_train_mae)
            history['val_mae'].append(avg_val_mae)
            
            # 保存最佳模型
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model(epoch, avg_val_loss, avg_val_mae)
                no_improve = 0
            else:
                no_improve += 1
            
            # 早停
            if no_improve >= early_stopping:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
        
        # 保存训练历史
        self.save_history(history)
        return history
    
    def save_model(self, epoch, val_loss, val_mae):
        """保存模型和相关信息"""
        try:
            # 使用Path对象处理路径
            model_path = Path(r"F:\files\code\DL\dcn_model.pth")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型参数
            torch.save(self.model.state_dict(), str(model_path))
            print(f"模型已保存到: {model_path}")
            
            # 保存模型信息
            info_path = model_path.parent / "model_info.json"
            model_info = {
                'epoch': epoch,
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'input_dim': self.model.cross_layers[0].input_dim,
                'architecture': {
                    'num_cross_layers': len(self.model.cross_layers),
                    'deep_layers': [layer.out_features for layer in self.model.deep_layers if isinstance(layer, nn.Linear)]
                }
            }
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
                
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
    
    def save_history(self, history):
        """保存训练历史"""
        history_path = self.save_dir / "training_history.json"
        
        # 转换numpy数组为列表
        history = {k: [float(v) for v in vals] for k, vals in history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

def save_model(model, scaler, save_dir):
    """保存模型权重和归一化器"""
    # 提取模型权重
    model_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer = {'weight': param.detach().cpu().numpy()}
        elif 'bias' in name:
            layer['bias'] = param.detach().cpu().numpy()
            model_weights.append(layer)
    
    # 保存权重和归一化器
    with open(save_dir / 'model_weights.pkl', 'wb') as f:
        pickle.dump(model_weights, f)
    
    joblib.dump(scaler, save_dir / 'scaler.pkl')

def train_model(model, train_loader, val_loader, num_epochs=200, device='cuda'):
    # 优化器设置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # 使用自适应学习率调度器
    scheduler = AdaptiveLRScheduler(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        warmup_epochs=5,
        cooldown_epochs=3
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    # 记录最佳模型和指标
    best_model = None
    best_metrics = {
        'epoch': 0,
        'val_loss': float('inf'),
        'val_r2': float('-inf'),
        'val_mae': float('inf'),
        'train_loss': None,
        'train_r2': None,
        'train_mae': None
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_metrics = {'r2': 0, 'mae': 0, 'rmse': 0}
        
        # 训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (features, targets) in enumerate(train_pbar):
            features, targets = features.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = model.custom_loss(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 计算指标
            batch_metrics = calculate_metrics(outputs, targets)
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]
            
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.2e}',
                'r2': f'{batch_metrics["r2"]:.3f}',
                'mae': f'{batch_metrics["mae"]:.3f}'
            })
        
        # 计算平均训练指标
        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_metrics = {'r2': 0, 'mae': 0, 'rmse': 0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            for features, targets in val_pbar:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = model.custom_loss(outputs, targets)
                
                # 计算指标
                batch_metrics = calculate_metrics(outputs, targets)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]
                
                val_loss += loss.item()
                
                # 更新进度条
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.2e}',
                    'r2': f'{batch_metrics["r2"]:.3f}',
                    'mae': f'{batch_metrics["mae"]:.3f}'
                })
        
        # 计算平均验证指标
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # 打印epoch结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, R²: {train_metrics["r2"]:.4f}, MAE: {train_metrics["mae"]:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, R²: {val_metrics["r2"]:.4f}, MAE: {val_metrics["mae"]:.4f}')
        
        # 使用自适应学习率调度
        current_lr = scheduler.step(val_loss)
        print(f'Current learning rate: {current_lr:.6f}')
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.best_model)
            break
        
        # 检查是否是最佳模型
        if val_loss < best_metrics['val_loss']:
            best_metrics.update({
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_r2': val_metrics['r2'],
                'val_mae': val_metrics['mae'],
                'train_loss': train_loss,
                'train_r2': train_metrics['r2'],
                'train_mae': train_metrics['mae']
            })
            best_model = copy.deepcopy(model.state_dict())
            print("\n发现更好的模型!")
            print(f"最佳验证损失: {val_loss:.4f}")
            print(f"最佳验证R²: {val_metrics['r2']:.4f}")
            print(f"最佳验证MAE: {val_metrics['mae']:.4f}")
    
    # 训练结束后，返回最佳模型和指标
    model.load_state_dict(best_model)
    return model, best_metrics

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # 加载数据
        data_loader = HousePriceDataLoader()
        train_loader, val_loader, test_loader = data_loader.get_data_loaders(
            batch_size=64,
            train_split=0.8,
            val_split=0.1
        )
        
        # 获取输入维度
        input_dim = len(data_loader.get_feature_names())
        print(f"Input dimension: {input_dim}")
        
        # 创建模型
        model = create_dcn_model(input_dim=input_dim, device=device)
        print("Model created successfully")
        
        # 训练模型
        trained_model, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=200,
            device=device
        )
        
        print("\n训练完成!")
        print(f"最佳模型出现在第 {best_metrics['epoch']} 轮")
        print(f"最佳验证指标:")
        print(f"Loss: {best_metrics['val_loss']:.4f}")
        print(f"R²: {best_metrics['val_r2']:.4f}")
        print(f"MAE: {best_metrics['val_mae']:.4f}")
        
        # 保存最佳模型
        save_dir = Path(r"F:\files\code\DL")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        model_path = save_dir / "best_dcn_model.pth"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'metrics': best_metrics,
            'model_config': {
                'input_dim': input_dim,
                'num_cross_layers': 4,
                'deep_layers': [256, 128, 64],
                'dropout': 0.3
            }
        }, model_path)
        print(f"\n最佳模型已保存到: {model_path}")
        
        # 保存归一化器
        save_model(trained_model, data_loader.scaler_x, save_dir)
        print("模型和归一化器保存成功")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")