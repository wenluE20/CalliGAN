import argparse
import cv2
import os, random, glob
from pathlib import Path
import sys
from PIL import Image, ImageEnhance, ImageFont
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.preprocessing_helper import (
    CANVAS_SIZE,
    CHAR_SIZE,
    draw_example_src_only,
    draw_single_char,
    draw_single_char_by_font,
)
# 不导入需要缺失文件的模块


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired grayscale images for training and evaluation. "
            "Defaults point to the staged archive data so the script can be run "
            "without manual arguments (useful in PowerShell)."
        )
    )

    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "img_folder",
        nargs="?",
        default=str(repo_root / "prepared_data" / "data" / "train"),
        help="Root folder containing input calligraphy images.",
    )
    parser.add_argument(
        "dst_folder_all",
        nargs="?",
        default=str(repo_root / "img_all"),
        help="Output directory for train/test splits of paired images.",
    )
    parser.add_argument(
        "dst_folder_cns",
        nargs="?",
        default=str(repo_root / "img_all_cns"),
        help="Output directory for CNS-coded paired images.",
    )
    return parser.parse_args()
# 使用项目根目录中的字体文件
src_font = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simkai.ttf")
print(f"使用字体文件: {src_font}")

def get_char(folder_name):
    char_list = []
    for filename in os.listdir(folder_name):
        # 支持jpg和png格式
        if '.jpg' in filename.lower() or '.png' in filename.lower():
            # 从文件名中提取第一个字符
            char = filename[0]
            char_list.append(char)
    return char_list


def get_union_and_intersect(img_folder, folder_list):
    # 获取所有存在的文件夹
    existing_folders = []
    for folder in folder_list:
        folder_path = os.path.join(img_folder, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            existing_folders.append(folder)
    
    print(f"找到 {len(existing_folders)} 个存在的文件夹")
    
    # 计算所有文件夹中的字符集合
    char_sets = []
    for folder in existing_folders:
        chars = get_char(os.path.join(img_folder, folder))
        if chars:  # 确保文件夹不为空
            char_sets.append(set(chars))
    
    if not char_sets:
        print("警告: 没有找到包含PNG文件的文件夹")
        return [], []
    
    # 计算并集和交集
    union_set = char_sets[0].copy()
    for char_set in char_sets[1:]:
        union_set.update(char_set)
    
    intersect_set = char_sets[0].copy()
    for char_set in char_sets[1:]:
        intersect_set.intersection_update(char_set)
    
    return list(union_set), list(intersect_set)

"""
def getSharedCharacter(folder_list, img_folder):
    charDict = dict()
    sharedChar = []
    tmpChar = []
    unionChar = []
    intersectChar = []
    # filenameList = []
    for idx, folder in enumerate(folder_list):
        for filename in os.listdir(os.path.join(img_folder, folder)):
            if '.png' in filename:
                char = filename[0]
                tmpChar.append(char)

                if idx == 0 and char not in charDict:
                    charDict[char] = 1
                elif idx != 0 and char in charDict:
                    charDict[char] += 1
                else:
                    continue

    for idx, (key, value) in enumerate(charDict.items()):
        if value == 7:
            sharedChar.append(key)

    return sharedChar
"""


def select_test_character(intersect_list):
    # 动态调整测试字符数量
    sample_size = min(1000, len(intersect_list))
    if sample_size == 0:
        print("警告: 交集列表为空，无法选择测试字符")
        return []
    return random.sample(intersect_list, sample_size)


# output image file name: [category]_[count].jpg
def generatePairImg(selectedTestChar, save_folder_all, save_folder_cns, folder_list, img_folder):
    train = save_folder_all + '/train/'
    test = save_folder_all + '/test/'
    train_cns = save_folder_cns + '/train/'
    test_cns = save_folder_cns + '/test/'
    if not os.path.exists(train):
        os.makedirs(train)
    if not os.path.exists(test):
        os.makedirs(test)
    if not os.path.exists(train_cns):
        os.makedirs(train_cns)
    if not os.path.exists(test_cns):
        os.makedirs(test_cns)

    font = ImageFont.truetype(src_font, CHAR_SIZE)
    count_test = 1
    count_train = 1

    # 只处理存在的文件夹
    existing_folders = []
    for idx, folder in enumerate(folder_list):
        folder_path = os.path.join(img_folder, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            existing_folders.append((idx, folder))
    
    print(f"正在处理 {len(existing_folders)} 个存在的文件夹")

    for idx, folder in existing_folders:
        src_folder = os.path.join(img_folder, folder)
        print(f"处理文件夹: {src_folder}")
        # 搜索jpg和png文件
        jpg_files = glob.glob(os.path.join(src_folder, '*.jpg'))
        png_files = glob.glob(os.path.join(src_folder, '*.png'))
        all_files = jpg_files + png_files
        
        for path in all_files:
            filename = path[len(src_folder)+1:]
            substr = str(filename[0])
                # 跳过字符组件分析，直接处理图像
            component = substr  # 使用字符本身作为组件标识，避免依赖缺失的文件

            if component is None:
                    component = "unknown"  # 使用默认值避免None值错误

            try:
                image = Image.open(path)
                # read calligraphy image and modify size
                calli_img = draw_single_char(image, canvas_size=CANVAS_SIZE, char_size=CHAR_SIZE)
                # Add contrast
                contrast = ImageEnhance.Contrast(calli_img)
                calli_img = contrast.enhance(2.)
                # Add brightness
                brightness = ImageEnhance.Brightness(calli_img)
                calli_img = brightness.enhance(2.)

                #get corresponding font image
                #font_img = draw_single_char_by_font(substr, font, CANVAS_SIZE, CHAR_SIZE)
                #im_AB = np.concatenate([font_img, char_img], 1)
                together = draw_example_src_only(substr, font, calli_img, CANVAS_SIZE, CHAR_SIZE)

                if substr in selectedTestChar:
                    together.save(os.path.join(test, "%d_%d.jpg" %(idx, count_test)))
                    together.save(os.path.join(test_cns, "%s_%d_%d.jpg" %(str(component), idx, count_test))) # 确保component是字符串类型
                    count_test += 1
                else:
                    together.save(os.path.join(train, "%d_%d.jpg" %(idx, count_train)))
                    together.save(os.path.join(train_cns, "%s_%d_%d.jpg" %(str(component), idx, count_train))) # 确保component是字符串类型
                    count_train += 1

            except OSError:
                with open(save_folder_all + '/error_msg.txt', 'a') as f:
                    f.write("cannot open image file %s \n" %(filename))
            except IOError:
                with open(save_folder_all + '/error_msg.txt', 'a') as f:
                    f.write("cannot open image file %s \n" %(filename))


def main():
    args = parse_args()
    img_folder_path = Path(args.img_folder)
    if not img_folder_path.exists():
        raise FileNotFoundError(
            f"Input folder not found: {img_folder_path}. "
            "Stage the archive data with crawler/crawler.py or pass the correct path explicitly."
        )

    img_folder = str(img_folder_path)
    dst_folder_all = args.dst_folder_all
    dst_folder_cns = args.dst_folder_cns

    folder_list = ['bdsr', 'csl', 'fwq', 'gj', 'htj', 'hy', 'lgq', 'lqs', 'lx', 'mf', 'mzd', 'oyx', 'sgt', 'shz', 'smh', 'wxz', 'wzm', 'yyr', 'yzq', 'zmf']
    union_list, intersect_list = get_union_and_intersect(img_folder, folder_list)
    print("union_list len: ", len(union_list))
    # union_list len:  6548
    print("intersect_list len: ", len(intersect_list))
    # intersect_list len:  5560

    # sharedChar = getSharedCharacter(folder_list, img_folder)
    # print(len(sharedChar)) 3857
    generatePairImg(selectedTestChar=select_test_character(intersect_list), save_folder_all=dst_folder_all, save_folder_cns=dst_folder_cns, folder_list=folder_list, img_folder=img_folder)


if __name__ == '__main__':
    main()