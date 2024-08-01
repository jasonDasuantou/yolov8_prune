# yolov8模型剪枝项目

## 项目简介
本项目通过应用模型剪枝技术，旨在降低深度学习模型的复杂性和计算负载，并通过回调训练进一步提升模型的效率和性能。

## 功能特点
- **模型剪枝**：去除冗余权重，精简模型结构。
- **回调训练**：剪枝后对模型进行再训练，优化性能。
- **性能优化**：在减小模型体积的同时，保持或提高模型的准确性和泛化能力。

## 使用技术
- 深度学习框架：PyTorch
- 配置和权重文件：用于模型定义和初始化。
- Python脚本：自定义脚本进行模型训练和调整。

## 运行环境
- Python 3.8
- 深度学习库：PyTorch
- CUDA环境（推荐，用于GPU加速）

## 安装指南
1. 克隆项目仓库到本地机器
   ```bash
   git clone https://github.com/jasonDasuantou/yolov8_prune.git
   python train_step1.py
