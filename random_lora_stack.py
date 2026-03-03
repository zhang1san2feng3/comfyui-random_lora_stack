import os
import random
from PIL import Image
import numpy as np
import re
import torch

def get_lora_folders():
    """获取ComfyUI/models/loras下的所有文件夹路径"""
    try:
        current_file = os.path.abspath(__file__)
        comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        loras_full_path = os.path.join(comfyui_dir, "models", "loras")
        
        if not os.path.exists(loras_full_path):
            return [""]
        
        folders = [""]
        for root, dirs, files in os.walk(loras_full_path):
            for dir_name in dirs:
                full_dir_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(full_dir_path, loras_full_path)
                folders.append(rel_path)
        
        return [""] + sorted([f for f in folders if f != ""])
    except Exception:
        return [""]

def find_content_by_tag(full_text, tag_name):
    """内部工具：寻找指定标签的内容"""
    if not tag_name or tag_name.strip() == "":
        return None
    start_pattern = re.compile(re.escape(tag_name.strip()) + r"\s*[:：]", flags=re.MULTILINE)
    start_match = start_pattern.search(full_text)
    if not start_match:
        return None
    content_start = start_match.end()
    next_tag_pattern = re.compile(r"^[^\r\n:：]+\s*[:：]", flags=re.MULTILINE)
    next_match = next_tag_pattern.search(full_text, pos=content_start)
    if next_match:
        return full_text[content_start:next_match.start()].strip()
    else:
        return full_text[content_start:].strip()

def extract_tag_section(full_text: str, tag_name: str) -> str:
    """按三级降级逻辑提取提示词"""
    if not full_text:
        return ""
    result = find_content_by_tag(full_text, tag_name)
    if result is not None:
        return result
    if tag_name.strip() != "默认服装":
        result = find_content_by_tag(full_text, "默认服装")
        if result is not None:
            return result
    return full_text

class RandomLoraStackLoader:
    @classmethod
    def INPUT_TYPES(cls):
        lora_folders = get_lora_folders()
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lora_folder1": (lora_folders, {"default": ""}),
                "tag_name1": ("STRING", {"default": "默认服装"}),
                "weight1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable1": ("BOOLEAN", {"default": True}),
                "lora_folder2": (lora_folders, {"default": ""}),
                "tag_name2": ("STRING", {"default": "默认服装"}),
                "weight2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable2": ("BOOLEAN", {"default": True}),
                "lora_folder3": (lora_folders, {"default": ""}),
                "tag_name3": ("STRING", {"default": "默认服装"}),
                "weight3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable3": ("BOOLEAN", {"default": True}),
                "lora_folder4": (lora_folders, {"default": ""}),
                "tag_name4": ("STRING", {"default": "默认服装"}),
                "weight4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable4": ("BOOLEAN", {"default": True}),
                "lora_folder5": (lora_folders, {"default": ""}),
                "tag_name5": ("STRING", {"default": "默认服装"}),
                "weight5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable5": ("BOOLEAN", {"default": True}),
                "lora_folder6": (lora_folders, {"default": ""}),
                "tag_name6": ("STRING", {"default": "默认服装"}),
                "weight6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "enable6": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_stack_in": ("LORA_STACK",),
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
                          lora_folder1, tag_name1, weight1, enable1,
                          lora_folder2, tag_name2, weight2, enable2,
                          lora_folder3, tag_name3, weight3, enable3,
                          lora_folder4, tag_name4, weight4, enable4,
                          lora_folder5, tag_name5, weight5, enable5,
                          lora_folder6, tag_name6, weight6, enable6,
                          lora_stack_in=None):
        random.seed(seed)
        
        folders = [lora_folder1, lora_folder2, lora_folder3, lora_folder4, lora_folder5, lora_folder6]
        tag_names = [tag_name1, tag_name2, tag_name3, tag_name4, tag_name5, tag_name6]
        weights = [weight1, weight2, weight3, weight4, weight5, weight6]
        enables = [enable1, enable2, enable3, enable4, enable5, enable6]
        
        out_display_names = [] # 这里存储最终显示的名称（可能是##后的，也可能是文件名）
        out_images = []
        out_texts = []
        lora_stack_out = list(lora_stack_in) if lora_stack_in else []

        current_file = os.path.abspath(__file__)
        comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        loras_base_path = os.path.join(comfyui_dir, "models", "loras")

        for i in range(6):
            if not enables[i] or not folders[i]:
                out_display_names.append("未启用")
                out_images.append(torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1)
                out_texts.append("")
                continue

            full_folder_path = os.path.join(loras_base_path, folders[i])
            if not os.path.exists(full_folder_path):
                out_display_names.append("文件夹不存在")
                out_images.append(torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1)
                out_texts.append("")
                continue

            all_files = os.listdir(full_folder_path)
            lora_files = [f for f in all_files if f.lower().endswith((".safetensors", ".ckpt"))]
            
            if not lora_files:
                out_display_names.append("无LORA文件")
                out_images.append(torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1)
                out_texts.append("")
                continue

            sel_lora = random.choice(lora_files)
            base_name = os.path.splitext(sel_lora)[0]
            
            # 默认显示名称为文件名
            final_display_name = sel_lora 

            # 存入 LORA 堆栈
            rel_lora_path = os.path.join(folders[i], sel_lora)
            lora_stack_out.append((rel_lora_path, weights[i], weights[i]))

            # 图片预览逻辑
            img_found = False
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                img_path = os.path.join(full_folder_path, base_name + ext)
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_np = np.array(img).astype(np.float32) / 255.0
                        out_images.append(torch.from_numpy(img_np).unsqueeze(0))
                        img_found = True
                        break
                    except: pass
            if not img_found:
                out_images.append(torch.rand(1, 64, 64, 3, dtype=torch.float32) * 0.1)

            # 核心修改：文本读取逻辑与名称别名提取
            txt_path = os.path.join(full_folder_path, base_name + ".txt")
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        raw_text = "".join(lines)
                        
                        # --- 检查第一行是否有 ## 别名 ---
                        if lines and lines[0].strip().startswith("##"):
                            # 提取 ## 之后的内容，去掉前后的空格和换行
                            final_display_name = lines[0].strip()[2:].strip()
                    
                    # 使用提取函数获取提示词
                    final_text = extract_tag_section(raw_text, tag_names[i])
                    out_texts.append(final_text)
                except:
                    out_texts.append("TXT读取失败")
            else:
                out_texts.append("")
            
            out_display_names.append(final_display_name)

        return (lora_stack_out, *out_display_names, *out_images, *out_texts)

NODE_CLASS_MAPPINGS = {"随机LORA堆栈加载器": RandomLoraStackLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"随机LORA堆栈加载器": "随机LORA堆栈加载器"}
