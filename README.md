# CNN vs MLP 实验项目

## 数据集路径说明
MNIST 和 CIFAR-100 数据集将自动下载至 `./data/` 文件夹。

## 环境依赖
见 `requirements.txt`，推荐使用 Python 3.9。

## 启动示例
```bash
python main_mlp_mnist.py
python main_cnn_mnist.py
python main_cnn_cifar.py
```

## 下面是一个满足你实验要求的完整 代码项目结构说明 + 各模块功能概览，包括：

MLP 和 CNN 对比实验（MNIST）

自定义 CNN 结构训练（CIFAR-100）

使用 Thop 计算参数量和计算量

使用 Grad-CAM 可视化模型注意区域

使用 Matplotlib 可视化训练过程

附加 README 示例和建议结构图工具

## 项目结构
<pre><code class="language-bash">
test1/
│
├── models/
│   ├── mlp.py              # MLP模型定义（用于MNIST）
│   ├── cnn_mnist.py        # CNN模型定义（用于MNIST）
│   └── cnn_cifar.py        # 自定义CNN（CIFAR-100）
│
├── utils/
│   ├── train_eval.py       # 通用训练和评估函数
│   ├── analysis.py         # Thop计算、GradCAM可视化
│   └── plot.py             # 绘图函数：loss曲线、性能分析
│
├── data/
│   └── transforms.py       # 数据预处理（MNIST + CIFAR-100）
│
├── main_mlp_mnist.py       # MLP训练主函数
├── main_cnn_mnist.py       # CNN训练主函数
├── main_cnn_cifar.py       # CIFAR-100 CNN 主训练文件
│
├── requirements.txt        # 所需环境
├── README.md               # 使用说明 & 数据集预处理说明
└── result/                 # 训练日志、模型权重、图像输出
</code></pre>