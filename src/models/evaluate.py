import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import sys
# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parent.parent.parent
# project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
from src.models.deep_learning.dcn_model import create_dcn_model
from src.models.model_utils.data_loader import HousePriceDataLoader

class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = Path(model_path)
        self.load_model()
        
    def load_model(self):
        """加载模型和配置"""
        checkpoint = torch.load(self.model_path)
        model_config = checkpoint['model_config']
        
        # 创建模型
        self.model = create_dcn_model(
            input_dim=model_config['input_dim'],
            device=self.device
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 保存训练指标
        self.training_metrics = checkpoint['metrics']
        
    def evaluate(self, test_loader):
        """评估模型性能"""
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, target in test_loader:
                features = features.to(self.device)
                output = self.model(features)
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算各种指标
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'mape': np.mean(np.abs((targets - predictions) / targets)) * 100
        }
        
        return predictions, targets, metrics
    
    def plot_predictions(self, predictions, targets, save_dir=None):
        """Plot prediction vs actual values"""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.scatter(targets, predictions, alpha=0.5, label='Predictions')
        
        # Ideal line
        min_val = min(min(predictions), min(targets))
        max_val = max(max(predictions), max(targets))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Line')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.legend()
        
        if save_dir:
            plt.savefig(save_dir / 'predictions_vs_targets.png')
        plt.show()
        
    def plot_error_distribution(self, predictions, targets, save_dir=None):
        """Plot error distribution"""
        errors = predictions - targets
        
        plt.figure(figsize=(12, 6))
        
        # Error histogram
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        
        if save_dir:
            plt.savefig(save_dir / 'error_distribution.png')
        plt.show()
        
    def plot_residuals(self, predictions, targets, save_dir=None):
        """Plot residuals"""
        residuals = predictions - targets
        
        plt.figure(figsize=(12, 6))
        plt.scatter(predictions, residuals, alpha=0.5, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.legend()
        
        if save_dir:
            plt.savefig(save_dir / 'residuals.png')
        plt.show()
        
    def save_evaluation_report(self, metrics, save_dir):
        """保存评估报告"""
        report = {
            'test_metrics': metrics,
            'training_metrics': self.training_metrics
        }
        
        with open(save_dir / 'evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

def main():
    # 设置路径
    model_path = Path(r"F:\files\code\DL\best_dcn_model.pth")
    save_dir = Path(r"F:\files\code\DL\evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # 加载数据
        data_loader = HousePriceDataLoader()
        _, _, test_loader = data_loader.get_data_loaders(
            batch_size=64,
            train_split=0.8,
            val_split=0.1
        )
        
        # 创建评估器
        evaluator = ModelEvaluator(model_path, device)
        
        # 评估模型
        predictions, targets, metrics = evaluator.evaluate(test_loader)
        
        # 打印评估指标
        print("\nModel Evaluation Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        # 设置matplotlib的全局字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        # 绘制可视化图表
        evaluator.plot_predictions(predictions, targets, save_dir)
        evaluator.plot_error_distribution(predictions, targets, save_dir)
        evaluator.plot_residuals(predictions, targets, save_dir)
        
        # 保存评估报告
        evaluator.save_evaluation_report(metrics, save_dir)
        print(f"\nEvaluation report saved to: {save_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        print(f"Detailed error message:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main() 