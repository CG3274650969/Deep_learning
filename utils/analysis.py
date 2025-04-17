import thop
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from thop import profile

# 用 Thop 分析模型的参数和计算量
def analyze_model(model, input_size=(1, 1, 28, 28), name="model"):
    model.eval()
    dummy_input = torch.randn(*input_size)

    # 确保 model 和 input 在同一个 device
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    print(f"Model {name}: FLOPs: {flops / 1e6:.2f} MFLOPs")
    print(f"Model {name}: Params: {params / 1e6:.2f} M")
    return flops, params

# 使用 GradCAM 可视化
def visualize_gradcam(model, input_image, target_layer):
    # 检查输入的图像维度并预处理
    input_image = preprocess_image(input_image)
    
    grad_cam = GradCAM(model, target_layer)
    
    # 获取热力图和目标图像
    mask, result = grad_cam(input_image)
    
    # 将热力图应用到原图上
    result = show_cam_on_image(input_image[0].cpu().numpy(), result[0].cpu().numpy(), use_rgb=True)
    
    # 可视化结果
    plt.imshow(result)
    plt.show()