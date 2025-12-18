# -*- coding: utf-8 -*-
import pickle
import os

def verify_pickle_file(file_path):
    """验证pickle文件的内容和格式"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return False
    
    print(f"正在验证文件: {file_path}")
    try:
        count = 0
        with open(file_path, 'rb') as f:
            while True:
                try:
                    example = pickle.load(f)
                    count += 1
                    # 检查数据格式是否正确
                    if len(example) != 3:
                        print(f"错误: 第{count}条数据格式不正确")
                        return False
                    cns_code, label, img_bytes = example
                    # 检查字段类型
                    if not isinstance(cns_code, str):
                        print(f"错误: CNS代码不是字符串类型")
                        return False
                    if not isinstance(label, int):
                        print(f"错误: 标签不是整数类型")
                        return False
                    if not isinstance(img_bytes, bytes):
                        print(f"错误: 图像数据不是字节类型")
                        return False
                    # 显示前几个样本的信息
                    if count <= 5:
                        print(f"样本 {count}: CNS代码='{cns_code}', 标签={label}, 图像大小={len(img_bytes)}字节")
                except EOFError:
                    break
        
        print(f"验证成功! 找到 {count} 个样本")
        return count > 0
    except Exception as e:
        print(f"验证过程中出错: {e}")
        return False

if __name__ == "__main__":
    train_file = "prepared_data/data/cns_train.obj"
    test_file = "prepared_data/data/cns_test.obj"
    
    print("=== 验证训练集数据 ===")
    train_valid = verify_pickle_file(train_file)
    
    print("\n=== 验证测试集数据 ===")
    test_valid = verify_pickle_file(test_file)
    
    if train_valid and test_valid:
        print("\n✓ 所有数据文件验证通过!")
    else:
        print("\n✗ 数据验证失败!")
