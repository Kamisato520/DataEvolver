#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像整理脚本
将ObjMove-A文件夹中的图像按类型整理到objectmoverA文件夹中
"""

import glob
import os
import shutil

from PIL import Image


def create_directories():
    """创建目标文件夹结构"""
    base_dir = "/huggingface/dataset_hub/ObjectMover-Benchmark/objectmoverB"

    # 创建主文件夹
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"创建主文件夹: {base_dir}")

    # 创建8个子文件夹
    subdirs = [
        "object",
        "images",
        "images_resize",
        "source_mask",
        "source_mask_resize",
        "target_bb",
        "target_bb_resize",
    ]

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f"创建子文件夹: {subdir_path}")


def convert_to_png(input_path, output_path):
    """将图像转换为PNG格式"""
    try:
        with Image.open(input_path) as img:
            # 如果是RGBA模式，保持透明通道
            if img.mode in ("RGBA", "LA"):
                img.save(output_path, "PNG")
            else:
                # 转换为RGB模式
                rgb_img = img.convert("RGB")
                rgb_img.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"转换图像失败 {input_path}: {e}")
        return False


def organize_images():
    """整理图像文件"""
    source_dir = "/huggingface/dataset_hub/ObjectMover-Benchmark/ObjMove-B"
    target_dir = "/huggingface/dataset_hub/ObjectMover-Benchmark/objectmoverB"

    # 文件映射关系
    file_mapping = {
        "src_object_crop.png": "object",
        "src_input.png": "images",
        "src_input_resized.png": "images_resize",
        "src_mask_hr.png": "source_mask",
        "src_mask_hr_resized.png": "source_mask_resize",
        "tar_box_mask.png": "target_bb",
        "tar_box_mask_resized.png": "target_bb_resize",
    }

    # 获取所有real_*文件夹
    real_dirs = glob.glob(os.path.join(source_dir, "test_*"))
    real_dirs.sort()  # 按名称排序

    print(f"找到 {len(real_dirs)} 个子文件夹")

    for real_dir in real_dirs:
        folder_name = os.path.basename(real_dir)
        print(f"处理文件夹: {folder_name}")

        for source_file, target_subdir in file_mapping.items():
            source_path = os.path.join(real_dir, source_file)
            target_path = os.path.join(target_dir, target_subdir, f"{folder_name}.png")

            if os.path.exists(source_path):
                # 复制并转换图像
                if convert_to_png(source_path, target_path):
                    print(f"  ✓ {source_file} -> {target_subdir}/{folder_name}.png")
                else:
                    print(f"  ✗ 处理失败: {source_file}")
            else:
                print(f"  ! 文件不存在: {source_file}")


def main():
    """主函数"""
    print("开始整理图像文件...")

    # 检查源文件夹是否存在
    if not os.path.exists("/huggingface/dataset_hub/ObjectMover-Benchmark/ObjMove-B"):
        print("错误: ObjMove-B 文件夹不存在!")
        return

    # 创建目录结构
    create_directories()

    # 整理图像
    organize_images()

    print("图像整理完成!")


if __name__ == "__main__":
    main()
