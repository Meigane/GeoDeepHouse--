import torch
import numpy as np
from sklearn.metrics import r2_score
import copy

class EarlyStopping:
    """改进的早停机制"""
    def __init__(self, patience=7, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            
class AdaptiveLRScheduler:
    """自适应学习率调度器"""
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, 
                 min_lr=1e-6, warmup_epochs=5, cooldown_epochs=3):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.bad_epochs = 0
        self.cooldown_counter = 0
        self.current_epoch = 0
        
        # 保存初始学习率
        self.initial_lr = optimizer.param_groups[0]['lr']
        
    def step(self, metric):
        """更新学习率"""
        self.current_epoch += 1
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Warmup阶段
        if self.current_epoch <= self.warmup_epochs:
            lr = self.initial_lr * (self.current_epoch / self.warmup_epochs)
            self._set_lr(lr)
            print(f"\nWarmup阶段 - 学习率调整为: {lr:.6f}")
            return current_lr
            
        # Cooldown阶段
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            print(f"\nCooldown阶段 - 剩余{self.cooldown_counter}个epoch")
            return current_lr
            
        # 检查是否需要调整学习率
        improved = (self.mode == 'min' and metric < self.best_metric) or \
                  (self.mode == 'max' and metric > self.best_metric)
                  
        if improved:
            self.best_metric = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            
        # 如果连续多个epoch没有改善，降低学习率
        if self.bad_epochs >= self.patience:
            if current_lr > self.min_lr:
                new_lr = max(current_lr * self.factor, self.min_lr)
                self._set_lr(new_lr)
                print(f"\n性能未改善 {self.patience} 个epoch - 学习率降低到: {new_lr:.6f}")
                
                # 重置计数器并开始cooldown
                self.bad_epochs = 0
                self.cooldown_counter = self.cooldown_epochs
            else:
                print(f"\n学习率已达到最小值: {self.min_lr}")
                
        return self.optimizer.param_groups[0]['lr']
        
    def _set_lr(self, lr):
        """设置所有参数组的学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def calculate_metrics(pred, target):
    """计算多个评估指标"""
    with torch.no_grad():
        # 转换为numpy数组
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # 计算R方
        r2 = r2_score(target_np, pred_np)
        
        # 计算MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        } 