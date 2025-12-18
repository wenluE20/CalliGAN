import os
import glob
import argparse
from PIL import Image

# 复用你项目里已经写好的标准化与拼接逻辑
from preprocess.preprocessing_helper import draw_single_char, draw_paired_image, CANVAS_SIZE, CHAR_SIZE


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_ids(folder: str):
    # 只接受 4 位编号 + .jpg
    paths = glob.glob(os.path.join(folder, "[0-9][0-9][0-9][0-9].jpg"))
    ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in paths])
    return ids


def make_one_pair(src_path: str, dst_path: str) -> Image.Image:
    # 打开两张单字图
    src_img = Image.open(src_path)
    dst_img = Image.open(dst_path)

    # 标准化到 256×256 灰度单字图（helper 内部会 convert("L") 并裁剪/缩放）
    src_std = draw_single_char(src_img, canvas_size=CANVAS_SIZE, char_size=CHAR_SIZE)
    dst_std = draw_single_char(dst_img, canvas_size=CANVAS_SIZE, char_size=CHAR_SIZE)

    # 拼接成 512×256：左 dst，右 src（draw_paired_image 内部就是这样 paste 的）
    paired = draw_paired_image(src_std, dst_std, CANVAS_SIZE)
    return paired


def process_split(split: str, src_root: str, dst_root: str, out_root: str, strict: bool = True):
    src_dir = os.path.join(src_root, split)
    dst_dir = os.path.join(dst_root, split)
    out_dir = os.path.join(out_root, split)
    ensure_dir(out_dir)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Missing src split folder: {src_dir}")
    if not os.path.isdir(dst_dir):
        raise FileNotFoundError(f"Missing dst split folder: {dst_dir}")

    src_ids = set(list_ids(src_dir))
    dst_ids = set(list_ids(dst_dir))

    common_ids = sorted(src_ids & dst_ids)
    only_src = sorted(src_ids - dst_ids)
    only_dst = sorted(dst_ids - src_ids)

    if strict and (only_src or only_dst):
        msg = [
            f"[{split}] ID mismatch:",
            f"  src-only: {len(only_src)} (e.g. {only_src[:5]})",
            f"  dst-only: {len(only_dst)} (e.g. {only_dst[:5]})",
            "Fix the missing files or rerun with --non_strict to skip.",
        ]
        raise RuntimeError("\n".join(msg))

    # 非 strict 模式下，只处理交集
    error_log_path = os.path.join(out_root, f"errors_{split}.txt")
    err_count = 0

    for i, id_ in enumerate(common_ids, 1):
        src_path = os.path.join(src_dir, f"{id_}.jpg")
        dst_path = os.path.join(dst_dir, f"{id_}.jpg")
        out_path = os.path.join(out_dir, f"{id_}.jpg")

        try:
            paired = make_one_pair(src_path, dst_path)
            paired.save(out_path, "JPEG")
        except Exception as e:
            err_count += 1
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{id_}.jpg\t{type(e).__name__}: {e}\n")

        if i % 200 == 0 or i == len(common_ids):
            print(f"[{split}] {i}/{len(common_ids)} done. errors={err_count}")

    print(f"[{split}] finished. saved={len(common_ids)-err_count}, errors={err_count}")
    if not strict:
        print(f"[{split}] skipped (src-only)={len(only_src)}, (dst-only)={len(only_dst)}")


def main():
    parser = argparse.ArgumentParser(
        description="Make 512x256 paired images from src/dst single-character JPGs (ID-aligned)."
    )
    parser.add_argument("--src_root", required=True, help="Root of source font images (contains train/ and test/).")
    parser.add_argument("--dst_root", required=True, help="Root of target font images (contains train/ and test/).")
    parser.add_argument("--out_root", required=True, help="Output root for paired images (will create train/ test/).")
    parser.add_argument("--non_strict", action="store_true", help="Skip missing IDs instead of failing.")
    args = parser.parse_args()

    ensure_dir(args.out_root)

    strict = not args.non_strict
    process_split("train", args.src_root, args.dst_root, args.out_root, strict=strict)
    process_split("test", args.src_root, args.dst_root, args.out_root, strict=strict)


if __name__ == "__main__":
    main()
