import os
import glob
from PIL import Image

IN_DIR = r"prepared_data\infer"      # 你的原始 64x64 小图
OUT_DIR = r"prepared_data\infer_paired"  # 输出 512x256 拼接图

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")))
    if not paths:
        raise RuntimeError(f"No jpg files found in {IN_DIR}")

    for i, p in enumerate(paths, 1):
        src = Image.open(p).convert("L")
        # 先强制变成 256x256（先跑通，后面再追求更好裁剪）
        src256 = src.resize((256, 256), Image.BILINEAR)

        # 做 512x256 拼接图：左白右src
        paired = Image.new("L", (512, 256), 255)
        paired.paste(src256, (256, 0))   # 右半边放 source
        # 左半边保持白底（也可以 paired.paste(src256, (0,0)) 复制一份）

        out_path = os.path.join(OUT_DIR, os.path.basename(p))
        paired.save(out_path, "JPEG")

        if i % 200 == 0 or i == len(paths):
            print(f"{i}/{len(paths)} done")

if __name__ == "__main__":
    main()
