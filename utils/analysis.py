import thop
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 用 Thop 分析模型的参数和计算量
def analyze_model(model, input_size=(1, 1, 28, 28)):
    inputs = torch.randn(*input_size)
    flops, params = thop.profile(model, inputs=(inputs,))
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"Params: {params/1e6:.2f}M")
    return flops, params

# 使用 GradCAM 可视化
def visualize_gradcam(model, input_image, target_layer):
    grad_cam = GradCAM(model, target_layer)
    mask, result = grad_cam(input_image)
    
    # 可视化
    plt.imshow(result[0].cpu().numpy(), cmap='jet')
    plt.show()
