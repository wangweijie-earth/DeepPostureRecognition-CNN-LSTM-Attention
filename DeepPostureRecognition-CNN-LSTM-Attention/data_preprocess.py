"""
data_preprocess.py

用途：
- 遍历给定数据根目录下的 class0..class9 子文件夹，读取每个子文件夹中的 sample.csv（有表头 ch0..ch7，无时间列）
- 对每个 CSV 做滑动窗口（window=15, step=8），生成窗口数据 (n_windows, window, 8)
- 每个类保存为 classX_windows.npz，同时生成合并文件 all_windows.npz
- 生成 train/val/test 的索引划分（随机，seed 可控）并保存为 all_splits.npz
- 输出每类统计信息（样本数、时长、窗口数、占比）

说明：
- 如果你在 IDE 中直接按 Run（没有在命令行传入参数），脚本会自动将 save_combined 和 save_splits 设为 True，
  并直接在默认 out_dir（root/output_windows） 下生成 all_windows.npz 和 all_splits.npz。
- 如果你仍希望通过命令行控制，可以像之前一样使用 argparse 参数。
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="足底压力数据预处理（滑窗并保存 npz）")
    p.add_argument("--root", type=str,
                   default=r"F:\360MoveData\Users\wangweijie\Desktop\pythonProject\data",
                   help="数据根目录，含 class0..class9 子文件夹")
    p.add_argument("--classes", type=int, default=10, help="类别数（默认 10，对应 class0..class9）")
    p.add_argument("--sampling_rate", type=int, default=10, help="采样率，Hz")
    p.add_argument("--window_s", type=float, default=1.5, help="窗口长度，秒")
    p.add_argument("--overlap", type=float, default=0.5, help="窗口重叠比例（0-1）")
    p.add_argument("--out_dir", type=str, default=None, help="输出目录（默认在 root/output_windows）")
    p.add_argument("--save_combined", action="store_true", help="是否保存合并的 all_windows.npz（默认 False）")
    p.add_argument("--save_splits", action="store_true", help="是否保存 train/val/test 划分索引（默认 False）")
    p.add_argument("--seed", type=int, default=42, help="随机种子，用于划分")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    return p.parse_args()

def sliding_windows(data: np.ndarray, window: int, step: int):
    """
    data: (n_samples, n_channels)
    return: windows (n_windows, window, n_channels)
    """
    n = data.shape[0]
    starts = list(range(0, max(0, n - window + 1), step))
    win_list = []
    for s in starts:
        e = s + window
        if e <= n:
            win_list.append(data[s:e, :])
    if len(win_list) == 0:
        return np.zeros((0, window, data.shape[1]), dtype=np.float32)
    return np.stack(win_list, axis=0)

def process_one_file(csv_path: Path, window: int, step: int):
    # 读取 CSV（假设有表头 ch0..ch7），取前 8 列
    df = pd.read_csv(csv_path)
    # 若列名不是预期也按前8列取值
    arr = df.iloc[:, :8].to_numpy(dtype=np.float32)
    windows = sliding_windows(arr, window, step)
    return windows, arr.shape[0]

def preprocess(root: Path,
               classes_num: int = 10,
               sampling_rate: int = 10,
               window_s: float = 1.5,
               overlap: float = 0.5,
               out_dir: Path = None,
               save_combined: bool = True,
               save_splits: bool = True,
               seed: int = 42,
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15):
    """
    主要处理函数，可以被外部调用。
    """
    if out_dir is None:
        out_dir = root / "output_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [f"class{i}" for i in range(classes_num)]
    sampling_rate = int(sampling_rate)
    window = int(round(window_s * sampling_rate))
    step = int(round(window * (1 - overlap)))
    if step < 1:
        step = 1

    print(f"根目录: {root}")
    print(f"窗口: {window_s}s -> {window} 样本, 步长: {step}")
    print(f"输出目录: {out_dir}")
    print(f"save_combined={save_combined}, save_splits={save_splits}")

    summary = []
    all_data_list = []
    all_labels_list = []

    for i, cls in enumerate(classes):
        cls_folder = root / cls
        csv_file = cls_folder / "sample.csv"
        if not csv_file.exists():
            print(f"[WARN] 未找到 {csv_file}，跳过该类。")
            summary.append({"class": cls, "label": i, "status": "missing", "n_samples": 0, "duration_s": 0.0, "n_windows": 0})
            continue

        windows, n_samples = process_one_file(csv_file, window, step)
        n_windows = windows.shape[0]
        duration_s = n_samples / sampling_rate

        # labels 全为 i
        labels = np.full((n_windows,), i, dtype=np.int64)

        # 保存每类文件
        out_file = out_dir / f"{cls}_windows.npz"
        np.savez_compressed(out_file, data=windows, labels=labels,
                            sampling_rate=sampling_rate, window=window, step=step)
        print(f"[OK] {cls}: samples={n_samples}, duration={duration_s:.2f}s, windows={n_windows} -> {out_file}")

        summary.append({"class": cls, "label": i, "status": "ok", "n_samples": int(n_samples), "duration_s": float(duration_s), "n_windows": int(n_windows)})
        if n_windows > 0:
            all_data_list.append(windows)
            all_labels_list.append(labels)

    # 汇总表
    summary_df = pd.DataFrame(summary)
    print("\n=== Summary ===")
    print(summary_df)

    # 合并并保存（可选）
    if save_combined and len(all_data_list) > 0:
        all_data = np.concatenate(all_data_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        combined_file = out_dir / "all_windows.npz"
        np.savez_compressed(combined_file, data=all_data, labels=all_labels,
                            sampling_rate=sampling_rate, window=window, step=step)
        print(f"[OK] 已保存合并文件: {combined_file} (n={all_data.shape[0]})")
    else:
        all_data = None
        all_labels = None

    # 生成 train/val/test 划分并保存索引（若请求）
    if save_splits:
        # 如果 all_data 可用则按其长度划分；否则加载每个 class 的 npz 合并做划分
        if all_data is None:
            loaded_data = []
            loaded_labels = []
            for i, cls in enumerate(classes):
                path = out_dir / f"{cls}_windows.npz"
                if not path.exists():
                    continue
                arr = np.load(path)
                if arr["data"].shape[0] == 0:
                    continue
                loaded_data.append(arr["data"])
                loaded_labels.append(arr["labels"])
            if len(loaded_data) == 0:
                print("[WARN] 未能找到任何窗口数据，无法生成划分。")
                return
            all_data = np.concatenate(loaded_data, axis=0)
            all_labels = np.concatenate(loaded_labels, axis=0)

        # 随机打乱并划分
        rng = np.random.RandomState(seed)
        n_total = all_data.shape[0]
        indices = np.arange(n_total)
        rng.shuffle(indices)

        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = n_total - n_train - n_val
        if n_test < 0:
            n_test = max(0, n_total - n_train - n_val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:n_train + n_val + n_test]

        splits_file = out_dir / "all_splits.npz"
        np.savez_compressed(splits_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                            total=n_total, seed=seed,
                            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        print(f"[OK] 已保存划分索引: {splits_file} (total={n_total}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

    print("处理完成。")

def main():
    # 解析命令行参数
    args = parse_args()

    # 如果脚本被直接 Run（没有额外命令行参数），我们自动打开保存合并与划分的选项，方便在 IDE 中直接运行。
    # 判定方法：sys.argv 长度为 1（只有脚本名）时认为是 IDE Run。
    if len(sys.argv) == 1:
        print("[INFO] 检测到未通过命令行传参（可能在 IDE 中直接运行）。")
        print("[INFO] 自动启用 save_combined=True 与 save_splits=True，会生成 all_windows.npz 与 all_splits.npz。")
        args.save_combined = True
        args.save_splits = True

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else None

    preprocess(root=root,
               classes_num=args.classes,
               sampling_rate=args.sampling_rate,
               window_s=args.window_s,
               overlap=args.overlap,
               out_dir=out_dir,
               save_combined=args.save_combined,
               save_splits=args.save_splits,
               seed=args.seed,
               train_ratio=args.train_ratio,
               val_ratio=args.val_ratio,
               test_ratio=args.test_ratio)

if __name__ == "__main__":
    main()

