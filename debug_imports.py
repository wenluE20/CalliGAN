import os
import sys
import traceback

# 检查文件路径和权限
file_path = 'models/unet_onehot_cns_font_attention.py'
print(f"Checking file: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")
print(f"File size: {os.path.getsize(file_path)} bytes")

# 尝试读取文件的第一部分
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read(5000)  # 读取更多内容
    print("\nFile can be read successfully.")
    print(f"First 5000 chars length: {len(content)}")
    print("\nLast 100 chars of the read content:")
    print(repr(content[-100:]))  # 显示最后100个字符，看是否正常读取完

# 尝试以不同的方式导入
try:
    print("\nTrying to import with __import__...")
    module = __import__('models.unet_onehot_cns_font_attention', fromlist=['UNet'])
    print("Module imported successfully with __import__")
    
    try:
        UNet = getattr(module, 'UNet')
        print("UNet class found in module")
    except AttributeError as e:
        print(f"UNet class not found: {e}")
        
except Exception as e:
    print(f"Error importing with __import__: {e}")
    traceback.print_exc()

print("\nDebug completed.")