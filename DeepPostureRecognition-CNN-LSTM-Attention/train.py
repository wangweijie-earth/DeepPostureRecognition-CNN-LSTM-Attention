# train_cnn_lstm_attn_v2_with_attn_tsne_confmat_fixed_full_save_both_with_metrics.py
"""
完整脚本：CNN + LSTM + Attention (v2)
功能：
 - attention heatmaps
 - t-SNE (initial / final / both)
 - 混淆矩阵以真实类归一化并以百分比显示
 - 输出 Precision / Recall / F1（按类 + macro/micro/weighted）
 - 绘制多类 ROC 曲线并计算 AUC（one-vs-rest + micro-average）
默认：augment=True
修改训练轮数请更改 NUM_EPOCHS（脚本顶部）
"""

from pathlib import Path
import numpy as np, random, time, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------ 可调参数（修改这里） ------------------
OUTPUT_DIR = r"F:\360MoveData\Users\wangweijie\Desktop\pythonProject\data\output_windows"
NUM_EPOCHS = 100          # <-- 改这里控制 epoch 次数
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 5e-4
SEED = 42
PATIENCE = 100              # early stopping patience
AUGMENT = True
SAVE_MODEL = "best_model_cnn_lstm_attn_v2_aug.pth"
NUM_WORKERS = 0
ATTN_PER_CLASS = 2

DO_TSNE = True
TSNE_MODE = "both"        # "initial", "final", "both"
TSNE_ON = "all"            # "test","val","train","all"
TSNE_SAMPLE_LIMIT = 0      # 0 表示不抽样（使用全部）
TSNE_POINT_SIZE = 80
TSNE_DPI = 200

# ------------------- 数据增强 -------------------
_rng = np.random.RandomState(42)
def jitter(x, sigma=0.01):
    scale = np.max(np.abs(x), axis=0, keepdims=True); scale[scale==0]=1.0
    return x + _rng.normal(0, sigma, x.shape) * scale
def scaling(x, sigma=0.08): return x * _rng.normal(1.0, sigma, (1,x.shape[1]))
def time_shift(x, max_shift=2):
    s = _rng.randint(-max_shift, max_shift+1)
    if s==0: return x
    res = np.zeros_like(x)
    if s>0:
        res[s:]=x[:-s]; res[:s]=x[0:1]
    else:
        ss=-s; res[:-ss]=x[ss:]; res[-ss:]=x[-1:]
    return res
def channel_dropout(x, drop_prob=0.15):
    mask = _rng.rand(x.shape[1])>drop_prob
    return x * mask[np.newaxis,:]
def augment_window(x):
    y = x.copy()
    if _rng.rand()<0.4: y = jitter(y, sigma=0.01)
    if _rng.rand()<0.4: y = scaling(y, sigma=0.08)
    if _rng.rand()<0.25: y = time_shift(y, max_shift=2)
    if _rng.rand()<0.2: y = channel_dropout(y, drop_prob=0.15)
    return y

# ------------------- Dataset -------------------
class WindowsDataset(Dataset):
    def __init__(self, windows, labels, training=True, augment=True):
        self.windows=windows; self.labels=labels; self.training=training; self.augment=augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x = self.windows[idx]; y = int(self.labels[idx])
        if self.training and self.augment: x = augment_window(x)
        x_t = torch.from_numpy(x.astype(np.float32)).permute(1,0)  # (C,T)
        return x_t, y

# ------------------- Model（CNN -> LSTM -> Attention -> FC） -------------------
class TimeAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, h):
        u = torch.tanh(self.w(h))
        scores = self.v(u).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha.unsqueeze(-1)*h, dim=1)
        return context, alpha

class CNN_LSTM_Attn_v2(nn.Module):
    def __init__(self, in_channels=8, n_classes=10, cnn_channels=[32,64], lstm_hidden=64, lstm_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        convs=[]; prev=in_channels
        for oc in cnn_channels:
            convs += [nn.Conv1d(prev, oc, kernel_size=3, padding=1), nn.BatchNorm1d(oc), nn.ReLU(), nn.MaxPool1d(2)]
            prev = oc
        self.cnn = nn.Sequential(*convs)
        self.bidirectional = bidirectional
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(input_size=prev, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        attn_in = lstm_hidden * (2 if bidirectional else 1)
        self.attn = TimeAttention(attn_in)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(attn_in, attn_in//2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(attn_in//2, n_classes))
    def forward(self, x):
        feat = self.cnn(x)            # (B, F, T')
        feat = feat.permute(0,2,1)    # (B, T', F)
        h, _ = self.lstm(feat)        # (B, T', H*dir)
        context, alpha = self.attn(h) # (B, H*dir), (B, T')
        logits = self.classifier(context)
        return logits, alpha
    def extract_context(self, x):
        feat = self.cnn(x); feat = feat.permute(0,2,1)
        h, _ = self.lstm(feat)
        context, alpha = self.attn(h)
        return context, alpha

# ------------------- 工具函数 -------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_windows(output_dir: Path):
    allf = output_dir / "all_windows.npz"
    if allf.exists():
        arr = np.load(allf, allow_pickle=True)
        return arr["data"], arr["labels"]
    datas=[]; labels=[]
    # 这里默认 class0..class9
    for i in range(10):
        p = output_dir / f"class{i}_windows.npz"
        if not p.exists(): continue
        arr = np.load(p, allow_pickle=True)
        if arr["data"].shape[0]==0: continue
        datas.append(arr["data"]); labels.append(arr["labels"])
    if len(datas)==0:
        raise RuntimeError("未找到任何 windows 数据，请先生成 classX_windows.npz 或 all_windows.npz")
    return np.concatenate(datas,0), np.concatenate(labels,0)

def plot_curves(logs, fname):
    epoch = range(1, len(logs["train_loss"])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epoch, logs["train_loss"], label="train_loss"); plt.plot(epoch, logs["val_loss"], label="val_loss"); plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(epoch, logs["train_acc"], label="train_acc"); plt.plot(epoch, logs["val_acc"], label="val_acc"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout(); plt.savefig(str(fname)); plt.close()

def plot_confusion_matrix_percent(y_true, y_pred, out_png, classes=None, dpi=150):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_perc = cm / row_sums * 100.0  # 百分比
    fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    n = cm.shape[0]
    ticks = range(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if classes is not None:
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
    else:
        ax.set_xticklabels([str(i) for i in ticks])
        ax.set_yticklabels([str(i) for i in ticks])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm_perc.shape[0]):
        for j in range(cm_perc.shape[1]):
            txt = f"{cm_perc[i, j]:.1f}%"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_perc[i, j] > cm_perc.max()/2 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=dpi); plt.close()
    print("Saved confusion matrix (percent) to", out_png)

def save_attention_heatmap(attn, sample_signal, out_path_png, out_path_npy=None, title=None, channel_plot=False):
    attn = np.array(attn).squeeze()
    Tprime = attn.shape[0]
    fig = plt.figure(figsize=(6,4 if not channel_plot else 4))
    if channel_plot:
        gs = fig.add_gridspec(2,1, height_ratios=(1,3))
        ax0 = fig.add_subplot(gs[0])
        sig = sample_signal  # (T, C)
        T = sig.shape[0]
        if T != Tprime:
            x_old = np.linspace(0,1,T); x_new = np.linspace(0,1,Tprime)
            sig_ds = np.zeros((Tprime, sig.shape[1]))
            for c in range(sig.shape[1]):
                sig_ds[:,c] = np.interp(x_new, x_old, sig[:,c])
        else:
            sig_ds = sig
        nshow = min(3, sig_ds.shape[1])
        for c in range(nshow):
            s = sig_ds[:,c]; s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            ax0.plot(np.arange(Tprime), s + c*1.2, label=f"ch{c}")
        ax0.set_yticks([]); ax0.set_xlim(0, Tprime-1); ax0.legend(loc="upper right", fontsize=8)
        ax1 = fig.add_subplot(gs[1])
    else:
        ax1 = fig.add_subplot(1,1,1)
    im = ax1.imshow(attn[np.newaxis, :], aspect='auto', cmap='viridis')
    ax1.set_yticks([]); ax1.set_xlabel("time step (reduced T')")
    if title: ax1.set_title(title)
    fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.05)
    plt.tight_layout(); plt.savefig(str(out_path_png)); plt.close()
    if out_path_npy is not None: np.save(str(out_path_npy), attn)

def tsne_initial_plot(data_all, labels_all, out_png, sample_limit=None, random_state=42, point_size=TSNE_POINT_SIZE, dpi=TSNE_DPI, show_plot=False):
    print("Running initial t-SNE...")
    N = data_all.shape[0]
    if sample_limit is not None and N > sample_limit:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, sample_limit, replace=False)
        X = data_all[idx]; y = labels_all[idx]
    else:
        X = data_all; y = labels_all
    X_flat = X.reshape(X.shape[0], -1)
    ts = TSNE(n_components=2, init='pca', random_state=random_state)
    X_tsne = ts.fit_transform(X_flat)
    plt.figure(figsize=(8,8), dpi=dpi)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=point_size, cmap='tab10', alpha=0.9, edgecolors='k', linewidths=0.2)
    plt.colorbar()
    plt.title("t-SNE initial (raw flattened windows)")
    plt.tight_layout(); plt.savefig(str(out_png), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()
    print("Saved initial t-SNE to", out_png)

def tsne_final_plot(model, data_all, labels_all, out_png, device, sample_limit=None, random_state=42, batch_size=64, point_size=TSNE_POINT_SIZE, dpi=TSNE_DPI, show_plot=False):
    print("Running final t-SNE (model embeddings)...")
    model.eval()
    N = data_all.shape[0]
    if sample_limit is not None and N > sample_limit:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, sample_limit, replace=False)
        labels_sampled = labels_all[idx]
    else:
        idx = np.arange(N); labels_sampled = labels_all
    contexts = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch = torch.from_numpy(data_all[batch_idx].astype(np.float32)).permute(0,2,1).to(device)  # (B, C, T)
            context, _ = model.extract_context(batch)
            contexts.append(context.cpu().numpy())
    contexts = np.concatenate(contexts, axis=0)
    ts = TSNE(n_components=2, init='pca', random_state=random_state)
    X_tsne = ts.fit_transform(contexts)
    plt.figure(figsize=(8,8), dpi=dpi)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels_sampled, s=point_size, cmap='tab10', alpha=0.9, edgecolors='k', linewidths=0.2)
    plt.colorbar()
    plt.title("t-SNE final (model context embeddings)")
    plt.tight_layout(); plt.savefig(str(out_png), dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()
    print("Saved final t-SNE to", out_png)

# ------------------- 新增：从 loader 获取概率输出 -------------------
def get_probs(model, loader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            ps.append(probs)
            ys.append(yb.numpy())
    if len(ps)==0:
        return np.array([]), np.array([])
    probs_all = np.concatenate(ps, axis=0)
    ys_all = np.concatenate(ys, axis=0)
    preds_all = probs_all.argmax(axis=1)
    return ys_all.tolist(), preds_all.tolist(), probs_all

# ------------------- ROC/AUC 绘图 -------------------
def plot_multiclass_roc(y_true, y_scores, n_classes, out_png, dpi=150):
    """
    y_true: array-like shape (N,)
    y_scores: array-like shape (N, n_classes) predicted probabilities
    """
    y_true = np.array(y_true)
    # binarize
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    # prepare figure
    plt.figure(figsize=(8,8), dpi=dpi)
    # per-class ROC
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        # If class i does not appear in y_true, skip
        if np.sum(Y[:, i]) == 0:
            fpr[i], tpr[i], roc_auc[i] = None, None, float('nan')
            continue
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"Class {i} (AUC={roc_auc[i]:.3f})")
    # micro-average
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})", color='k', lw=2, linestyle='--')
    except Exception:
        roc_auc["micro"] = float('nan')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC (one-vs-rest)')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout(); plt.savefig(str(out_png), dpi=dpi); plt.close()
    print("Saved ROC plot to", out_png)
    return roc_auc

# ------------------- 训练/验证 -------------------
def train_epoch(model, loader, criterion, optimizer, device, clip=1.0):
    model.train()
    losses=[]; y_t=[]; y_p=[]
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1).detach().cpu().numpy()
        y_p.extend(preds.tolist()); y_t.extend(yb.detach().cpu().numpy().tolist())
    return np.mean(losses) if losses else 0.0, accuracy_score(y_t, y_p) if y_t else 0.0

def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses=[]; y_t=[]; y_p=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            preds = logits.argmax(1).detach().cpu().numpy()
            y_p.extend(preds.tolist()); y_t.extend(yb.detach().cpu().numpy().tolist())
    return np.mean(losses) if losses else 0.0, accuracy_score(y_t, y_p) if y_t else 0.0, y_t, y_p

# ------------------- 主流程 -------------------
def main():
    set_seed(SEED)
    output_dir = Path(OUTPUT_DIR); output_dir.mkdir(parents=True, exist_ok=True)
    attn_outdir = output_dir / "attn_samples"; attn_outdir.mkdir(parents=True, exist_ok=True)
    tsne_outdir = output_dir / "tsne"; tsne_outdir.mkdir(parents=True, exist_ok=True)

    data, labels = load_windows(output_dir)
    print("Loaded", data.shape, labels.shape)

    # initial t-SNE（可选）
    if DO_TSNE and TSNE_MODE in ("initial","both"):
        sample_limit = None if TSNE_SAMPLE_LIMIT == 0 else TSNE_SAMPLE_LIMIT
        tsne_initial_plot(data, labels, tsne_outdir / "tsne_initial.png", sample_limit=sample_limit, random_state=SEED, show_plot=False)

    n_classes = int(labels.max())+1
    classes = [f"class{i}" for i in range(n_classes)]

    # 生成或加载 split（如存在 all_splits.npz 则直接使用）
    splits = output_dir / "all_splits.npz"
    if splits.exists():
        s = np.load(splits); train_idx=s["train_idx"]; val_idx=s["val_idx"]; test_idx=s["test_idx"]
    else:
        idx = np.arange(len(labels))
        tr_idx, te_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=SEED)
        val_rel = 0.15/0.3
        val_idx, test_idx = train_test_split(te_idx, test_size=(1-val_rel), stratify=labels[te_idx], random_state=SEED)
        train_idx = tr_idx
        np.savez(splits, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    X_train, y_train = data[train_idx], labels[train_idx]
    X_val, y_val = data[val_idx], labels[val_idx]
    X_test, y_test = data[test_idx], labels[test_idx]

    train_ds = WindowsDataset(X_train, y_train, training=True, augment=AUGMENT)
    val_ds = WindowsDataset(X_val, y_val, training=False, augment=False)
    test_ds = WindowsDataset(X_test, y_test, training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_Attn_v2(in_channels=data.shape[2], n_classes=n_classes, lstm_hidden=64, bidirectional=True, dropout=0.4).to(device)

    class_counts = np.bincount(y_train, minlength=n_classes); class_counts[class_counts==0]=1
    class_weights = torch.tensor(1.0/class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val = 1e9; no_improve=0
    logs = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    print(f"Start training (augment={AUGMENT}) -> saving to {SAVE_MODEL}")
    for epoch in range(1, NUM_EPOCHS+1):
        t0=time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, clip=1.0)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        logs["train_loss"].append(tr_loss); logs["val_loss"].append(val_loss)
        logs["train_acc"].append(tr_acc); logs["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f} | time {time.time()-t0:.1f}s")

        if val_loss < best_val - 1e-6:
            best_val = val_loss; no_improve=0
            torch.save(model.state_dict(), SAVE_MODEL); print("Saved", SAVE_MODEL)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered (no improvement).")
                break

    # 保存训练曲线
    curves_file = Path(output_dir) / f"loss_acc_curves_cnn_lstm_attn_v2_{'aug' if AUGMENT else 'noaug'}.png"
    plot_curves(logs, curves_file)

    # 加载最佳模型并在测试上评估
    if Path(SAVE_MODEL).exists():
        model.load_state_dict(torch.load(SAVE_MODEL, map_location=device))
        print("Loaded best model from", SAVE_MODEL)
    else:
        print("Warning: best model file not found, using current weights.")

    # 先用 eval_epoch 获取测试集的 loss/acc/preds
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    print("Test loss:", test_loss, "Test acc:", test_acc)

    # ---------------- 新增：计算 Precision / Recall / F1 ----------------
    from sklearn.metrics import precision_recall_fscore_support
    y_true_arr = np.array(y_true); y_pred_arr = np.array(y_pred)
    per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(y_true_arr, y_pred_arr, labels=list(range(n_classes)), zero_division=0)
    # 打印并保存
    metrics_txt = output_dir / f"precision_recall_f1_per_class_{'aug' if AUGMENT else 'noaug'}.txt"
    with open(metrics_txt, "w") as f:
        header = "class,precision,recall,f1,support\n"
        f.write(header)
        for i in range(n_classes):
            line = f"{i},{per_class_prec[i]:.4f},{per_class_rec[i]:.4f},{per_class_f1[i]:.4f},{support[i]}\n"
            f.write(line)
    # 也打印 macro/micro/weighted
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='micro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    summary_txt = output_dir / f"precision_recall_f1_summary_{'aug' if AUGMENT else 'noaug'}.txt"
    with open(summary_txt, "w") as f:
        f.write(f"Test acc: {test_acc:.6f}\n")
        f.write(f"Macro P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}\n")
        f.write(f"Micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}\n")
        f.write(f"Weighted P/R/F1: {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}\n")
    print("Saved precision/recall/f1 per-class to", metrics_txt)
    print("Saved precision/recall/f1 summary to", summary_txt)

    # 保存混淆矩阵（百分比形式）
    cm_file = Path(output_dir) / f"confusion_test_cnn_lstm_attn_v2_{'aug' if AUGMENT else 'noaug'}.png"
    plot_confusion_matrix_percent(y_true, y_pred, cm_file, classes=classes, dpi=TSNE_DPI)

    # ---------------- 保存 attention 热图 ----------------
    print("Saving attention heatmaps (per-class)...")
    model.eval()
    class_to_indices = {i: [] for i in range(n_classes)}
    for local_idx, global_idx in enumerate(test_idx):
        lbl = int(labels[global_idx])
        class_to_indices[lbl].append(global_idx)
    saved_count = 0
    with torch.no_grad():
        for cls_id, g_indices in class_to_indices.items():
            if len(g_indices) == 0: continue
            sel = g_indices[:ATTN_PER_CLASS] if len(g_indices) <= ATTN_PER_CLASS else list(np.random.RandomState(0).choice(g_indices, ATTN_PER_CLASS, replace=False))
            for gidx in sel:
                x = data[gidx]
                x_t = torch.from_numpy(x.astype(np.float32)).permute(1,0).unsqueeze(0).to(device)
                logits, attn = model(x_t)
                pred = int(logits.argmax(dim=1).cpu().numpy()[0])
                attn_np = attn.squeeze(0).cpu().numpy()
                png_name = attn_outdir / f"attn_class{cls_id}_idx{gidx}_pred{pred}.png"
                npy_name = attn_outdir / f"attn_class{cls_id}_idx{gidx}_pred{pred}.npy"
                save_attention_heatmap(attn_np, x, png_name, npy_name, title=f"class{cls_id} idx{gidx} pred{pred}", channel_plot=True)
                saved_count += 1
    print(f"Saved {saved_count} attention visualizations to {attn_outdir}")

    # ---------------- 可选 t-SNE（final / initial / both） ----------------
    if DO_TSNE:
        sample_limit = None if TSNE_SAMPLE_LIMIT == 0 else TSNE_SAMPLE_LIMIT

        if TSNE_MODE == "initial":
            tsne_initial_plot(data, labels, tsne_outdir / "tsne_initial.png", sample_limit=sample_limit, random_state=SEED, show_plot=False)

        elif TSNE_MODE == "final":
            # use chosen subset (TSNE_ON) for final plotting
            if TSNE_ON == "all":
                tsne_data = data; tsne_labels = labels
            elif TSNE_ON == "train":
                tsne_data = data[train_idx]; tsne_labels = labels[train_idx]
            elif TSNE_ON == "val":
                tsne_data = data[val_idx]; tsne_labels = labels[val_idx]
            else:
                tsne_data = data[test_idx]; tsne_labels = labels[test_idx]

            tsne_final_plot(model, tsne_data, tsne_labels, tsne_outdir / f"tsne_final_{TSNE_ON}.png", device,
                            sample_limit=sample_limit, random_state=SEED, batch_size=64, point_size=TSNE_POINT_SIZE, dpi=TSNE_DPI, show_plot=False)

        elif TSNE_MODE == "both":
            # initial: use full data by default
            tsne_initial_plot(data, labels, tsne_outdir / "tsne_initial.png", sample_limit=sample_limit, random_state=SEED, show_plot=False)

            # final: use TSNE_ON selection
            if TSNE_ON == "all":
                tsne_data = data; tsne_labels = labels
            elif TSNE_ON == "train":
                tsne_data = data[train_idx]; tsne_labels = labels[train_idx]
            elif TSNE_ON == "val":
                tsne_data = data[val_idx]; tsne_labels = labels[val_idx]
            else:
                tsne_data = data[test_idx]; tsne_labels = labels[test_idx]

            tsne_final_plot(model, tsne_data, tsne_labels, tsne_outdir / f"tsne_final_{TSNE_ON}.png", device,
                            sample_limit=sample_limit, random_state=SEED, batch_size=64, point_size=TSNE_POINT_SIZE, dpi=TSNE_DPI, show_plot=False)

            # 合并两张图并保存
            init_path = tsne_outdir / "tsne_initial.png"
            final_path = tsne_outdir / f"tsne_final_{TSNE_ON}.png"
            combined_path = tsne_outdir / f"tsne_initial_final_compare_{TSNE_ON}.png"
            if init_path.exists() and final_path.exists():
                img1 = mpimg.imread(str(init_path))
                img2 = mpimg.imread(str(final_path))
                fig, axes = plt.subplots(1,2,figsize=(16,8), dpi=TSNE_DPI)
                axes[0].imshow(img1); axes[0].set_title("Initial t-SNE"); axes[0].axis('off')
                axes[1].imshow(img2); axes[1].set_title("Final t-SNE"); axes[1].axis('off')
                plt.tight_layout(); plt.savefig(str(combined_path), dpi=TSNE_DPI); plt.close()
                print("Saved combined initial+final t-SNE to", combined_path)
            else:
                print("Warning: one of t-SNE files missing; cannot create combined image.")

    # ---------------- 新增：计算并绘制多类 ROC/AUC ----------------
    # 需要 y_true (list), y_pred (list), 以及 y_scores (NxC)
    y_true_list, y_pred_list, y_scores = get_probs(model, test_loader, device)
    if y_scores.size == 0:
        print("Warning: no scores obtained for ROC (empty). Skipping ROC plot.")
    else:
        roc_out = output_dir / f"roc_auc_{'aug' if AUGMENT else 'noaug'}.png"
        roc_auc_dict = plot_multiclass_roc(y_true_list, y_scores, n_classes, roc_out, dpi=TSNE_DPI)
        # 保存 AUC 值到文本
        auc_txt = output_dir / f"roc_auc_values_{'aug' if AUGMENT else 'noaug'}.txt"
        with open(auc_txt, "w") as f:
            for k,v in roc_auc_dict.items():
                f.write(f"{k}:{v}\n")
        print("Saved AUC values to", auc_txt)

    print("All done. Outputs saved to", output_dir)

if __name__ == "__main__":
    main()
