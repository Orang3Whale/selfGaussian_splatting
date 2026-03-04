#本代码旨在根据dino特征的差异，生成掩码
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image

RAW_FOLDER = "data_process\\dataset\\renders\\1027_raw"
RENDER_FOLDER = "data_process\\dataset\\renders\\1027_render"
RENDER_FOLDER_7000 = "data_process\\dataset\\renders\\3_4_7000"
def load_and_preprocess_image(image_path, patch_size=14, max_size=420):
    """
    加载图像，并限制最大分辨率以防止显存溢出，最后裁剪为 patch_size 的整数倍。
    """
    img = Image.open(image_path).convert('RGB')
    
    # === 核心防爆修改：等比例缩放限制最大边长 ===
    # 论文中设定为 350，我们这里可以稍微放宽到 420 (patch_size的整数倍)，保留更多一点细节
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        # 使用 LANCZOS 滤波器进行高质量降采样
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # ============================================

    w, h = img.size
    new_w, new_h = w - w % patch_size, h - h % patch_size
    img = img.crop((0, 0, new_w, new_h))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0), np.array(img), (new_w, new_h)
def main():
    print("正在加载 DINOv2 (ViT-S/14 Reg) 模型...")
    # 使用带寄存器(register)版本的 DINOv2，它对特征图中的伪影/高亮点抑制效果更好
    print("正在加载 DINOv2 模型架构...")
    # 1. pretrained=False 表示只拉取网络结构代码，不自动下载云端权重
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
    
    # 2. 指定你的本地 .pth 文件路径
    local_pth_path = "pth\\dinov2_vits14_reg4_pretrain.pth" 
    
    print(f"正在从本地注入权重: {local_pth_path}")
    # 3. 加载本地权重字典 (先加载到 CPU 内存，防止 GPU 显存溢出，之后再 .to(device))
    state_dict = torch.load(local_pth_path, map_location='cpu')
    
    # DINOv2 官方的 pth 文件有时会多包一层，比如 state_dict 可能会在 'model' 键下
    # 我们做一个安全检查：
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    # 4. 将权重注入模型空壳
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ================= 1. 替换为你的实际图片路径 =================
    path_original = os.path.join(RAW_FOLDER, 'input_0120.jpg')  # 真实的含噪图（有清晰行人）
    path_rendered = os.path.join(RENDER_FOLDER_7000, 'frame_00120.jpg')  # ns-render导出的图（行人变鬼影/透出背景）
    
    tensor_orig, img_orig_np, (W, H) = load_and_preprocess_image(path_original)
    tensor_rend, img_rend_np, _ = load_and_preprocess_image(path_rendered)
    
    tensor_orig = tensor_orig.to(device)
    tensor_rend = tensor_rend.to(device)
    
    patch_h, patch_w = H // 14, W // 14

    # ================= 2. 提取特征与计算余弦相似度 =================
    print("正在提取高维特征并计算空间一致性...")
    with torch.no_grad():
        feat_orig = model.forward_features(tensor_orig)['x_norm_patchtokens']
        feat_rend = model.forward_features(tensor_rend)['x_norm_patchtokens']
        
    # 计算余弦相似度并重塑为 2D 空间特征图
    cos_sim = F.cosine_similarity(feat_orig, feat_rend, dim=2)
    sim_map = cos_sim.reshape(patch_h, patch_w).cpu().numpy()
    
    # 将特征图双线性插值上采样回原始图像分辨率
    sim_map_resized = cv2.resize(sim_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # ================= 3. 阈值分割与形态学处理 =================
    # 【核心调参区】根据热力图的颜色分布调整此阈值
    # 相似度低于此值的区域将被判定为动态干扰物
    THRESHOLD = 0.65 
    
    # 二值化：干扰物设为 0 (Nerfstudio 默认 0 为遮罩忽略区)，静态背景设为 255
    binary_mask = np.where(sim_map_resized < THRESHOLD, 0, 255).astype(np.uint8)
    
    # 形态学滤波：利用光电/图像处理基础消除 DINO patch 带来的块状锯齿
    # 先闭运算（填补干扰物内部的白色噪点漏洞），再开运算（抹去背景里零星的黑色误判噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    # ================= 4. 可视化分析面板 =================
    # ================= 4. 纯 OpenCV 结果拼接与保存 =================
    print("正在使用 OpenCV 生成可视化对比图...")
    
    # 1. 将原图和渲染图从 RGB 转为 BGR (OpenCV 默认格式)
    img_orig_bgr = cv2.cvtColor(img_orig_np, cv2.COLOR_RGB2BGR)
    img_rend_bgr = cv2.cvtColor(img_rend_np, cv2.COLOR_RGB2BGR)
    
    # 2. 将 0~1 的余弦相似度矩阵归一化到 0~255，并应用伪彩色热力图映射
    # 余弦相似度正常在 -1 到 1 之间，通常对于相似图片分布在 0.5 到 1.0
    sim_normalized = cv2.normalize(sim_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_bgr = cv2.applyColorMap(sim_normalized, cv2.COLORMAP_JET)
    
    # 3. 将单通道的黑白 Mask 转为三通道，以便与其他图片拼接
    mask_bgr = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
    
    # 4. 在图片上添加文字标签 (可选，为了查看方便)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_orig_bgr, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(img_rend_bgr, "Rendered", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(heatmap_bgr, "DINO Heatmap", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(mask_bgr, f"Mask (Thresh: {THRESHOLD})", (10, 30), font, 1, (0, 255, 0), 2)
    
    # 5. 图像拼接：上排是原图和渲染图，下排是热力图和Mask
    top_row = cv2.hconcat([img_orig_bgr, img_rend_bgr])
    bottom_row = cv2.hconcat([heatmap_bgr, mask_bgr])
    final_canvas = cv2.vconcat([top_row, bottom_row])
    
    # 6. 保存到本地目录
    output_filename = "dino_mask_validation.jpg"
    cv2.imwrite(output_filename, final_canvas)
    print(f"验证完成！结果已保存为 {output_filename}")

if __name__ == "__main__":
    main()