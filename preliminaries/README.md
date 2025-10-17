# 数学与数据基础

本项目是学习 AI 底层数学与数据处理基础过程中的实操笔记与代码实现，  
目标是将理论概念与工程实践结合，通过可视化与代码实验加深理解。记录在此便于后续扩展与重用 

This repository documents my hands-on exploration of mathematical and data foundations for AI —  
bridging theory and engineering through visualization and code experiments. Document it here for future scalability and reuse.

---

## Topics

### NumPy 实践 [NumPy in Action](numpy_ndarray.ipynb)
- **数组操作**：创建、重塑、索引与切片  
- **图像处理**：使用 NumPy 与 PIL/OpenCV 加载与操作图像  
- **通道分离**：提取与可视化 RGB 通道  
- **模型评估**：利用 `argmax`、`argsort` 分析预测结果  

### 张量 [Tensor](tensor.ipynb)
- 基本数学运算、广播、索引、切片、变形、内存节省和转换其他 Python 对象。

### Pandas 数据预处理 [Data Preprocessing with Pandas](pandas.ipynb)
- **数据加载**：从 CSV 文件中读取结构化数据  
- **缺失值处理**：均值填充与类别型缺失处理  
- **数据转换**：将 DataFrame 转换为 PyTorch 张量，连接 AI 模型训练流程

### 线性代数基础 [Linear Algebra Essentials](linear-algebra.ipynb)
- **标量 / 向量 / 矩阵 / 张量**：理解从零维到多维的数据结构  
- **张量操作**：创建、形状变换、索引与基本运算  
- **特殊运算**：转置、Hadamard 积、矩阵乘法  
- **降维操作**：求和与平均值计算  

**重要数学概念：**
- **范数（Norms）**
  - L1 范数：绝对值之和  
  - L2 范数：欧几里得距离  
  - Frobenius 范数：矩阵整体的能量度量  
- **点积（Dot Product）**：向量相似度计算的基础  
- **矩阵-向量积（Matrix-Vector Product）**：神经网络层计算的核心操作  

### 自动微分 [autograd](autograd.ipynb)
- **计算图概念**：PyTorch 使用有向无环图(DAG)跟踪张量操作，当张量设置 requires_grad=True 时，所有基于该张量的操作都会被记录
- **‌反向传播**：通过调用 backward() 函数自动计算梯度
- **关键特性**：梯度累积与清零、非标量反向传播、计算图分离 (detach())、控制流的梯度计算

### Torchvision 数据处理 [Data Preprocessing with Torchvision](torchvision.ipynb)
- **Dataset类**：抽象基类，用于表示数据集
- **DataLoader类**：迭代器，用于批量获取数据
- **Torchvision计算机视觉工具**：数据集处理、数据增强、图像变换、数据标准化
- **常见的模型**：
---

## Learning Approach
- **理论结合实践**：每个数学概念均配有代码示例  
- **可视化辅助理解**：使用 `matplotlib` 展示数据分布、矩阵操作效果   

---
