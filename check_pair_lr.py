import os
from PIL import Image

# 改成你的拼接图路径（train 里任意一张）
PAIR_PATH = r"prepared_data\paired_images\train\0001.jpg"
OUT_DIR = r"prepared_data\debug_pair"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    img = Image.open(PAIR_PATH).convert("L")
    w, h = img.size
    print("pair size:", (w, h))

    assert (w, h) == (512, 256), "这张不是 512x256 拼接图，请换一张"

    left = img.crop((0, 0, 256, 256))    # 左半
    right = img.crop((256, 0, 512, 256)) # 右半

    left.save(os.path.join(OUT_DIR, "left.jpg"))
    right.save(os.path.join(OUT_DIR, "right.jpg"))

    print("saved:", os.path.join(OUT_DIR, "left.jpg"))
    print("saved:", os.path.join(OUT_DIR, "right.jpg"))

if __name__ == "__main__":
    main()
