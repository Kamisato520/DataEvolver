#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片重命名脚本
去除文件名中的"result_"前缀
"""

import argparse
import glob
import os
import sys


def rename_images_in_folder(folder_path):
    """重命名指定文件夹内的图片文件"""
    print(f"处理文件夹: {folder_path}")

    # 获取所有以"result_"开头的图片文件
    pattern = os.path.join(folder_path, "result_*.png")
    image_files = glob.glob(pattern)

    # 也查找其他常见图片格式
    for ext in ["*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff"]:
        pattern = os.path.join(folder_path, f"result_{ext}")
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print(f"  在 {folder_path} 中未找到以'result_'开头的图片文件")
        return 0

    renamed_count = 0
    for old_path in image_files:
        # 获取文件名（不含路径）
        filename = os.path.basename(old_path)

        # 检查是否以"result_"开头
        if filename.startswith("result_"):
            # 去除"result_"前缀
            new_filename = filename[7:]  # 去掉前7个字符"result_"
            new_path = os.path.join(folder_path, new_filename)

            try:
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"  ✓ {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"  ✗ 重命名失败 {filename}: {e}")

    return renamed_count


def process_folder_recursively(folder_path):
    """递归处理文件夹及其子文件夹"""
    total_renamed = 0

    # 处理当前文件夹
    renamed_count = rename_images_in_folder(folder_path)
    total_renamed += renamed_count

    # 递归处理子文件夹
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                sub_renamed = process_folder_recursively(item_path)
                total_renamed += sub_renamed
    except PermissionError:
        print(f"  无法访问文件夹: {folder_path}")
    except Exception as e:
        print(f"  处理文件夹时出错 {folder_path}: {e}")

    return total_renamed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量重命名图片文件，去除"result_"前缀'
    )
    parser.add_argument(
        "folder_path",
        nargs="?",
        default="/huggingface/dataset_hub/ObjectMover-Benchmark/Results/objmove-B-our/removal",
        help="要处理的文件夹路径（如果不指定，则处理当前目录）",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="递归处理子文件夹"
    )

    args = parser.parse_args()

    print("开始批量重命名图片文件...")

    # 确定要处理的文件夹路径
    if args.folder_path:
        target_folder = os.path.abspath(args.folder_path)
        if not os.path.exists(target_folder):
            print(f"错误: 指定的文件夹不存在: {target_folder}")
            return
        if not os.path.isdir(target_folder):
            print(f"错误: 指定的路径不是文件夹: {target_folder}")
            return
    else:
        target_folder = os.getcwd()

    print(f"目标文件夹: {target_folder}")

    if args.recursive:
        print("递归处理子文件夹...")
        total_renamed = process_folder_recursively(target_folder)
    else:
        # 只处理指定文件夹
        total_renamed = rename_images_in_folder(target_folder)

    print(f"\n重命名完成! 总共重命名了 {total_renamed} 个文件")


if __name__ == "__main__":
    main()
