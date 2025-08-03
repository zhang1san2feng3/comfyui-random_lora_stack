import os
import random
from PIL import Image
import numpy as np

def get_lora_folders():
    """获取ComfyUI/models/loras下的所有文件夹路径（相对路径）"""
    try:
        # 获取当前文件的路径
        current_file = os.path.abspath(__file__)
        # 当前文件 -> comfyui-random_lora_stack文件夹 -> custom_nodes文件夹 -> ComfyUI文件夹
        comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        # ComfyUI/models/loras
        loras_full_path = os.path.join(comfyui_dir, "models", "loras")
        
        print(f"Searching for LORA folders in: {loras_full_path}")
        
        if not os.path.exists(loras_full_path):
            return [""]
        
        folders = [""]  # 空选项作为第一个选择
        
        # 递归获取所有文件夹
        for root, dirs, files in os.walk(loras_full_path):
            for dir_name in dirs:
                full_dir_path = os.path.join(root, dir_name)
                # 计算相对于loras目录的路径
                rel_path = os.path.relpath(full_dir_path, loras_full_path)
                folders.append(rel_path)
        
        # 按字母顺序排序，但保持空选项在最前面
        sorted_folders = [""] + sorted([f for f in folders if f != ""])
        return sorted_folders
    except Exception as e:
        return [""]

class RandomLoraStackLoader:
    """
    随机LORA堆栈加载器
    输入：6个LORA文件夹、6个权重、6个开关、随机种子
    输出：6个图片（原图）、6个文本（TXT内容），与输入文件夹一一对应，未启用或未选文件夹则不输出
    """
    @classmethod
    def INPUT_TYPES(cls):
        lora_folders = get_lora_folders()
        return {
            "optional": {
                "lora_stack_in": ("LORA_STACK",),
            },
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lora_folder1": (lora_folders, {"default": ""}),
                "weight1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable1": ("BOOLEAN", {"default": True}),
                "lora_folder2": (lora_folders, {"default": ""}),
                "weight2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable2": ("BOOLEAN", {"default": True}),
                "lora_folder3": (lora_folders, {"default": ""}),
                "weight3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable3": ("BOOLEAN", {"default": True}),
                "lora_folder4": (lora_folders, {"default": ""}),
                "weight4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable4": ("BOOLEAN", {"default": True}),
                "lora_folder5": (lora_folders, {"default": ""}),
                "weight5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable5": ("BOOLEAN", {"default": True}),
                "lora_folder6": (lora_folders, {"default": ""}),
                "weight6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable6": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LORA_STACK", 
                    "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
                    "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", 
                    "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack_out", 
                    "LORA名称1", "LORA名称2", "LORA名称3", "LORA名称4", "LORA名称5", "LORA名称6",
                    "预览图片1", "预览图片2", "预览图片3", "预览图片4", "预览图片5", "预览图片6", 
                    "文本内容1", "文本内容2", "文本内容3", "文本内容4", "文本内容5", "文本内容6")
    FUNCTION = "load_random_loras"
    CATEGORY = "LORA"

    def load_random_loras(self, seed, 
                          lora_folder1, weight1, enable1,
                          lora_folder2, weight2, enable2,
                          lora_folder3, weight3, enable3,
                          lora_folder4, weight4, enable4,
                          lora_folder5, weight5, enable5,
                          lora_folder6, weight6, enable6, lora_stack_in=None):
        random.seed(seed)
        folders = [lora_folder1, lora_folder2, lora_folder3, lora_folder4, lora_folder5, lora_folder6]
        weights = [weight1, weight2, weight3, weight4, weight5, weight6]
        enables = [enable1, enable2, enable3, enable4, enable5, enable6]
        images = []
        texts = []
        selected_loras = []  # 存储随机选择的LORA文件名
        
        # 获取完整的loras目录路径 - 使用相对路径方法
        current_file = os.path.abspath(__file__)
        comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        loras_base_path = os.path.join(comfyui_dir, "models", "loras")
        
        for idx in range(6):
            if not enables[idx] or not folders[idx]:
                images.append(None)
                texts.append("")
                selected_loras.append("未选择")
                continue
            
            # 将相对路径转换为完整路径
            folder = os.path.join(loras_base_path, folders[idx])
            
            if not os.path.isdir(folder):
                images.append(None)
                texts.append("")
                selected_loras.append("文件夹不存在")
                continue
                
            files = os.listdir(folder)
            
            # 获取所有LORA文件（.safetensors或.ckpt）
            lora_files = [f for f in files if f.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth"))]
            img_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
            txt_files = [f for f in files if f.lower().endswith(".txt")]
            
            if not lora_files:
                images.append(None)
                texts.append("")
                selected_loras.append("无LORA文件")
                continue
                
            # 随机选择一个LORA文件
            selected_lora = random.choice(lora_files)
            selected_loras.append(selected_lora)
            
            # 获取LORA文件名（不带后缀）
            lora_basename = os.path.splitext(selected_lora)[0]
            
            # 查找与LORA文件名匹配的图片和文本文件
            matching_img = None
            matching_txt = None
            
            for img_file in img_files:
                if os.path.splitext(img_file)[0] == lora_basename:
                    matching_img = img_file
                    break
                    
            for txt_file in txt_files:
                if os.path.splitext(txt_file)[0] == lora_basename:
                    matching_txt = txt_file
                    break
            
            # 处理图片文件
            if matching_img:
                img_path = os.path.join(folder, matching_img)
                try:
                    # 严格验证图片格式
                    img = Image.open(img_path)
                    # 检查是否为有效图片文件
                    img.verify()  # 验证图片完整性
                    img = Image.open(img_path)  # 重新打开（verify后需要重新打开）
                    img = img.convert('RGB')  # 强制转RGB
                    img_np = np.array(img)
                    # 严格检查shape：必须是3维且最后一维为3
                    if img_np.ndim == 3 and img_np.shape[2] == 3 and img_np.size > 0:
                        # 确保最小尺寸
                        h, w = img_np.shape[:2]
                        if h >= 1 and w >= 1:
                            images.append(img_np)
                        else:
                            images.append(np.zeros((64,64,3), dtype=np.uint8))
                    else:
                        images.append(np.zeros((64,64,3), dtype=np.uint8))
                except Exception as e:
                    images.append(np.zeros((64,64,3), dtype=np.uint8))
            else:
                # 没有对应图片，使用黑色占位图
                images.append(np.zeros((64,64,3), dtype=np.uint8))
            
            # 处理文本文件
            if matching_txt:
                txt_path = os.path.join(folder, matching_txt)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    texts.append(text)
                except Exception as e:
                    texts.append("无")
            else:
                # 没有对应文本文件
                texts.append("无")
        # 处理随机LORA名称输出
        processed_lora_names = [name if name else "未选择" for name in selected_loras]
        
        # 处理图片输出 - 创建标准的ComfyUI图片张量 (NHWC格式)
        import torch
        processed_images = []
        for idx, img in enumerate(images):
            if img is None:
                # 创建64x64的彩色噪声图片作为占位图，使用ComfyUI标准格式 [1, H, W, 3]
                noise_img = torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1  # 很暗的噪声
                processed_images.append(noise_img)
            else:
                # 处理有效图片
                try:
                    # 确保图片数据完整性
                    if img.shape[2] != 3:
                        raise ValueError(f"Invalid image channels: {img.shape[2]}")
                    
                    # 转换为float32并归一化到[0,1]
                    img_float = img.astype(np.float32) / 255.0
                    
                    # 转换为torch张量并保持 NHWC 格式 (H,W,C) -> (1,H,W,C)
                    tensor = torch.from_numpy(img_float)
                    tensor = tensor.unsqueeze(0)  # HWC -> 1HWC
                    
                    # 强制设置正确的数据类型
                    tensor = tensor.to(dtype=torch.float32)
                    tensor = tensor.contiguous()
                    
                    # 验证最终形状 - ComfyUI期望 [Batch, Height, Width, Channels]
                    assert tensor.dim() == 4, f"Tensor should be 4D, got {tensor.dim()}D"
                    assert tensor.shape[0] == 1, f"Batch size should be 1, got {tensor.shape[0]}"
                    assert tensor.shape[3] == 3, f"Channels should be 3, got {tensor.shape[3]}"
                    
                    processed_images.append(tensor)
                except Exception as e:
                    noise_img = torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1
                    processed_images.append(noise_img)
        
        # 处理文本输出
        processed_texts = [txt if txt else "无" for txt in texts]
        
        # 构建真正的LORA_STACK - 先添加输入的LORA堆栈，然后添加当前节点的LORA
        lora_stack_out = []
        
        # 如果有输入的LORA堆栈，先添加到输出
        if lora_stack_in is not None:
            lora_stack_out.extend(lora_stack_in)
        
        # 将当前节点选择的LORA添加到堆栈中
        for idx in range(6):
            if enables[idx] and folders[idx] and selected_loras[idx] not in ["未选择", "文件夹不存在", "无LORA文件"]:
                # 构建LORA文件的完整路径
                lora_file_path = os.path.join(folders[idx], selected_loras[idx])
                # 添加到LORA堆栈 - ComfyUI LORA_STACK格式: (lora_name, strength_model, strength_clip)
                lora_stack_out.append((lora_file_path, weights[idx], weights[idx]))
        
        # 确保返回的数据结构正确：1个lora_stack + 6个图片 + 6个文本
        # 创建全新的张量副本，彻底避免引用问题
        final_images = []
        for i in range(6):
            img = processed_images[i]
            # 创建完全独立的张量副本，保持NHWC格式
            new_tensor = torch.zeros_like(img, dtype=torch.float32)
            new_tensor.copy_(img)
            new_tensor = new_tensor.contiguous()
            
            # 最终验证 - ComfyUI图像格式: [Batch, Height, Width, Channels]
            if new_tensor.dim() != 4 or new_tensor.shape[0] != 1 or new_tensor.shape[3] != 3:
                new_tensor = torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1
            
            final_images.append(new_tensor)
        
        return (lora_stack_out, 
                processed_lora_names[0], processed_lora_names[1], processed_lora_names[2],
                processed_lora_names[3], processed_lora_names[4], processed_lora_names[5],
                final_images[0], final_images[1], final_images[2], 
                final_images[3], final_images[4], final_images[5],
                processed_texts[0], processed_texts[1], processed_texts[2],
                processed_texts[3], processed_texts[4], processed_texts[5])

NODE_CLASS_MAPPINGS = {
    "随机LORA堆栈加载器": RandomLoraStackLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "随机LORA堆栈加载器": "随机LORA堆栈加载器"
}
