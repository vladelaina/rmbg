from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
from pathlib import Path
import sys
import time

def print_progress_bar(current, total, start_time, bar_length=50):
    """打印字符版进度条和预计剩余时间（格式：HH:MM:SS）"""
    percent = float(current) * 100 / total
    arrow = '=' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    # 计算剩余时间
    elapsed_time = time.time() - start_time
    if current > 0:
        time_per_item = elapsed_time / current
        remaining_items = total - current
        remaining_seconds = time_per_item * remaining_items
        
        # 转换为时分秒格式
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        seconds = int(remaining_seconds % 60)
        time_str = f" - 预计剩余时间: {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        time_str = ""
    
    print(f'\r[{arrow}{spaces}] {percent:.2f}%{time_str}', end='', flush=True)

def process_image(model, image_path, output_path):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    
    image.save(output_path)

def main(input_path):
    # 初始化模型
    model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to('cuda')
    model.eval()

    input_path = Path(input_path).resolve()
    
    if input_path.is_dir():
        output_dir = input_path / 'no_bg_images'
        output_dir.mkdir(exist_ok=True)
        
        # 获取所有图片文件
        image_files = [f for f in input_path.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        total_files = len(image_files)
        
        # 记录开始时间
        start_time = time.time()
        
        # 处理每个图片并更新进度条
        for i, img_file in enumerate(image_files, 1):
            output_path = output_dir / img_file.name
            process_image(model, str(img_file), str(output_path))
            print_progress_bar(i, total_files, start_time)
        
        print()  # 打印一个换行
    
    elif input_path.is_file():
        output_path = input_path.parent / f'no_bg_{input_path.name}'
        process_image(model, str(input_path), str(output_path))
        print(f"已处理: {input_path.name}")
    
    else:
        print(f"输入路径无效: {input_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供图片路径或文件夹路径")
        print("使用方法: python try.py <path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    main(input_path)
