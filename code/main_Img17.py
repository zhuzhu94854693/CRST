
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import math
import random
import shutil
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from tqdm import tqdm
from skimage import io
from skimage.filters.thresholding import threshold_otsu
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CRST_CD, CRST_CD_Loss
from dataset import OpticalSARChangeDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
seed_1 = 42
random.seed(seed_1)
np.random.seed(seed_1)
torch.manual_seed(seed_1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_1)
    torch.cuda.manual_seed_all(seed_1)

parser = argparse.ArgumentParser(
    description="CRST-CD",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--dataset", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--K", type=int, default=5,
                    help="Number of semantic classes, including class 0")
parser.add_argument(
    "--result_folder", type=str,
    default="./result/Img17/CRST"
)
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()

DATASET_NAMES = [
    "shuguang", "Italy", "yellow_C", "yellow_D", "Img17",
]
if not 1 <= args.dataset <= len(DATASET_NAMES):
    raise ValueError(
        f"--dataset must be between 1 and {len(DATASET_NAMES)}"
    )
dataset_name = DATASET_NAMES[args.dataset - 1]

print(f"Dataset: {dataset_name}")
print(f"Device: {device}")
print(f"K: {args.K}")

train_data = OpticalSARChangeDataset(
    train=True,
    dataset_id=args.dataset,
    repeat=200,
    augment=False
)
test_data = OpticalSARChangeDataset(
    train=False,
    dataset_id=args.dataset
)

img_height = int(test_data.H)
img_width = int(test_data.W)
print(f"Evaluation image size: {img_height} x {img_width}")

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available()
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available()
)

model = CRST_CD(
    in_channels=3,
    emb_dim=args.emb_dim,
    ks=args.ks,
    num_classes=args.K
).to(device)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model parameters: {n_params:.3f} M")

criterion = CRST_CD_Loss().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

WARMUP_EPOCHS = 5

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, args.epoch - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_numpy_label(x):

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x


def compute_otsu_binary(change_prob):

    prob = np.nan_to_num(
        change_prob, nan=0.0, posinf=1.0, neginf=0.0
    )
    prob = np.clip(prob, 0.0, 1.0)
    prob_u8 = np.round(prob * 255.0).astype(np.uint8)

    if prob_u8.min() == prob_u8.max():
        threshold_u8 = int(prob_u8.min())
    else:
        threshold_u8 = int(threshold_otsu(prob_u8))

    binary = (prob_u8 >= threshold_u8).astype(np.uint8)
    return binary, threshold_u8 / 255.0


def make_valid_binary_gt(gt):

    gt = np.asarray(gt)
    gt_bin = (np.nan_to_num(gt, nan=0.0) > 0.5).astype(np.uint8)
    all_pixels = np.ones(gt.shape, dtype=bool)
    return gt_bin, all_pixels


def evaluate_binary_saved(pred, gt, valid_mask=None):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    if valid_mask is None:
        valid = np.ones(gt.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask).astype(bool)

    p = pred[valid]
    g = gt[valid]
    n = int(g.size)
    if n == 0:
        return None

    tp = int(np.sum(p & g))
    tn = int(np.sum(~p & ~g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    oa = (tp + tn) / n

    pe = (
            ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp))
            / float(n * n)
    )
    kappa = (oa - pe) / (1.0 - pe) if (1.0 - pe) > 1e-12 else 0.0

    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "OA": float(oa),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "IoU": float(iou),
        "Kappa": float(kappa),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Specificity": float(specificity),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "n_valid": n
    }

def shared_hungarian_mapping(pred1, pred2, gt1, gt2, K):

    p = np.concatenate([
        np.asarray(pred1).reshape(-1),
        np.asarray(pred2).reshape(-1)
    ]).astype(np.int64)
    g = np.concatenate([
        np.asarray(gt1).reshape(-1),
        np.asarray(gt2).reshape(-1)
    ]).astype(np.int64)

    if p.size == 0:
        return {}

    D = max(
        int(K),
        int(p.max()) + 1,
        int(g.max()) + 1
    )
    conf = np.zeros((D, D), dtype=np.int64)
    np.add.at(conf, (p, g), 1)

    if D > 1:
        conf[0, :] = 0
    rows, cols = linear_sum_assignment(-conf)
    return {
        int(r): int(c)
        for r, c in zip(rows, cols)
        if int(r) != 0
    }


def apply_mapping(pred, mapping):
    pred = np.asarray(pred).astype(np.int64)
    out = pred.copy()
    for source, target in mapping.items():
        out[pred == source] = target
    return out


def evaluate_semantic_saved(pred, gt, valid_mask=None, K=5):

    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shapes differ: {pred.shape} vs {gt.shape}")

    p = pred.reshape(-1)
    g = gt.reshape(-1)
    n = int(g.size)
    if n == 0:
        return None
    if np.any(p < 0) or np.any(g < 0):
        raise ValueError("Negative semantic labels found; remap labels before evaluation.")

    D = max(int(K), int(p.max()) + 1, int(g.max()) + 1)
    conf = np.zeros((D, D), dtype=np.int64)
    np.add.at(conf, (p, g), 1)

    oa = float(np.mean(p == g))
    rows, cols = linear_sum_assignment(-conf)
    clustering_acc = float(conf[rows, cols].sum() / max(n, 1))

    if len(np.unique(g)) > 1 and len(np.unique(p)) > 1:
        ari = float(adjusted_rand_score(g, p))
        nmi = float(normalized_mutual_info_score(g, p))
    else:
        ari = float("nan")
        nmi = float("nan")

    classes = sorted(int(c) for c in np.unique(g))
    per_class = {}
    f1s, ious = [], []
    for c in classes:
        tp = int(np.sum((p == c) & (g == c)))
        fp = int(np.sum((p == c) & (g != c)))
        fn = int(np.sum((p != c) & (g == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        per_class[c] = {"Precision": float(precision), "Recall": float(recall),
                        "F1": float(f1), "IoU": float(iou),
                        "TP": tp, "FP": fp, "FN": fn}
        f1s.append(f1)
        ious.append(iou)

    return {"OA": oa, "Hungarian_ACC": clustering_acc, "ARI": ari, "NMI": nmi,
            "mean_F1": float(np.mean(f1s)) if f1s else 0.0,
            "mIoU": float(np.mean(ious)) if ious else 0.0,
            "n": n, "classes": classes, "per_class": per_class}


def scd_metrics_saved(sem1_idx, sem2_idx, gt_t1, gt_t2, gt_change, K):

    sem1_idx = np.asarray(sem1_idx).astype(np.int64)
    sem2_idx = np.asarray(sem2_idx).astype(np.int64)
    gt_t1 = np.asarray(gt_t1).astype(np.int64)
    gt_t2 = np.asarray(gt_t2).astype(np.int64)
    gt_change = np.asarray(gt_change) > 0.5

    if not (sem1_idx.shape == sem2_idx.shape == gt_t1.shape == gt_t2.shape == gt_change.shape):
        raise ValueError("SCD prediction/label arrays must have identical shapes")

    p1 = sem1_idx.reshape(-1)
    p2 = sem2_idx.reshape(-1)
    g1 = gt_t1.reshape(-1)
    g2 = gt_t2.reshape(-1)
    gt_change_flat = gt_change.reshape(-1)
    n_valid = int(g1.size)
    if n_valid == 0:
        return None

    mapping = shared_hungarian_mapping(p1, p2, g1, g2, K)
    p1a = apply_mapping(p1, mapping)
    p2a = apply_mapping(p2, mapping)

    gt_change_sem = (g1 != g2)
    gt_change_bin = gt_change_flat
    mismatch = float(np.mean(gt_change_sem != gt_change_bin))

    pred_change = (p1a != p2a)

    tp_cd = int(np.sum(pred_change & gt_change_sem))
    fp_cd = int(np.sum(pred_change & ~gt_change_sem))
    fn_cd = int(np.sum(~pred_change & gt_change_sem))

    pre = tp_cd / (tp_cd + fp_cd) if tp_cd + fp_cd else 0.0
    rec = tp_cd / (tp_cd + fn_cd) if tp_cd + fn_cd else 0.0

    correct_both = (p1a == g1) & (p2a == g2)
    oa = float(np.mean(correct_both))

    tp_scd = int(np.sum(pred_change & gt_change_sem & correct_both))
    fp_scd = int(
        np.sum(pred_change & ~gt_change_sem)
        + np.sum(pred_change & gt_change_sem & ~correct_both)
    )
    fn_scd = int(
        np.sum(~pred_change & gt_change_sem)
        + np.sum(pred_change & gt_change_sem & ~correct_both)
    )

    pre_scd = tp_scd / (tp_scd + fp_scd) if tp_scd + fp_scd else 0.0
    rec_scd = tp_scd / (tp_scd + fn_scd) if tp_scd + fn_scd else 0.0
    fscd = (
        2 * pre_scd * rec_scd / (pre_scd + rec_scd)
        if pre_scd + rec_scd else 0.0
    )

    classes = sorted(set(np.unique(g1)) | set(np.unique(g2)))
    per_class = {}
    iou_values = []

    for c in classes:
        tp1 = int(np.sum((p1a == c) & (g1 == c)))
        fp1 = int(np.sum((p1a == c) & (g1 != c)))
        fn1 = int(np.sum((p1a != c) & (g1 == c)))
        den1 = tp1 + fp1 + fn1
        iou1 = tp1 / den1 if den1 else 0.0

        tp2 = int(np.sum((p2a == c) & (g2 == c)))
        fp2 = int(np.sum((p2a == c) & (g2 != c)))
        fn2 = int(np.sum((p2a != c) & (g2 == c)))
        den2 = tp2 + fp2 + fn2
        iou2 = tp2 / den2 if den2 else 0.0

        iou_c = 0.5 * (iou1 + iou2)
        iou_values.append(iou_c)
        per_class[int(c)] = {
            "IoU": float(iou_c),
            "IoU_T1": float(iou1),
            "IoU_T2": float(iou2),
            "TP_T1": tp1, "FP_T1": fp1, "FN_T1": fn1,
            "TP_T2": tp2, "FP_T2": fp2, "FN_T2": fn2
        }

    miou = float(np.mean(iou_values)) if iou_values else 0.0

    all_classes = sorted(
        set(np.unique(g1)) | set(np.unique(g2))
        | set(np.unique(p1a)) | set(np.unique(p2a))
    )
    pc = 0.0
    for c in all_classes:
        pc += 0.5 * float(np.mean(p1a == c)) * float(np.mean(g1 == c))
        pc += 0.5 * float(np.mean(p2a == c)) * float(np.mean(g2 == c))

    sek = (oa - pc) / (1.0 - pc) if 1.0 - pc > 1e-12 else 0.0
    score = fscd * sek

    return {
        "OA": float(oa),
        "Pre": float(pre),
        "Rec": float(rec),
        "Fscd": float(fscd),
        "mIoU": float(miou),
        "Sek": float(sek),
        "Score": float(score),
        "Pre_scd": float(pre_scd),
        "Rec_scd": float(rec_scd),
        "n_valid": n_valid,
        "gt_classes": [int(c) for c in classes],
        "per_class": per_class,
        "mapping": mapping,
        "gt_change_mismatch": mismatch,
        "TP_cd": tp_cd, "FP_cd": fp_cd, "FN_cd": fn_cd
    }


def train_one_epoch(loader, model, criterion, optimizer, epoch):
    model.train()
    loss_sum = 0.0
    n_batches = 0

    with tqdm(loader, desc=f"Train {epoch + 1}") as progress:
        for batch in progress:
            optical = batch["Im_optical"].to(device, non_blocking=True)
            sar = batch["Im_sar"].to(device, non_blocking=True)

            pseudo = batch["pseudo"].to(device, non_blocking=True)
            pseudo_l1 = batch["pseudo_l1"].to(device, non_blocking=True)
            pseudo_l2 = batch["pseudo_l2"].to(device, non_blocking=True)

            out = model(optical, sar)
            loss = criterion(out, pseudo, pseudo_l1, pseudo_l2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.detach().item())
            n_batches += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

    return loss_sum / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(loader, model):
    model.eval()
    K = args.K

    canvas_change = np.zeros((img_height, img_width), dtype=np.float32)
    canvas_gt_change = np.zeros((img_height, img_width), dtype=np.float32)
    canvas_q1 = np.zeros((K, img_height, img_width), dtype=np.float32)
    canvas_q2 = np.zeros((K, img_height, img_width), dtype=np.float32)
    canvas_l1 = np.zeros((img_height, img_width), dtype=np.int64)
    canvas_l2 = np.zeros((img_height, img_width), dtype=np.int64)

    s_sum, d_sum, n_batches = 0.0, 0.0, 0

    for batch in loader:
        optical = batch["Im_optical"].to(device, non_blocking=True)
        sar = batch["Im_sar"].to(device, non_blocking=True)
        out = model(optical, sar)

        s_sum += float(out["s"].mean().item())
        d_sum += float(out["d_cross"].mean().item())
        n_batches += 1

        change_t = out["change_prob"]
        q1_t = out["q1_logits_gated"]
        q2_t = out["q2_logits_gated"]
        if change_t.shape[-2:] != (img_height, img_width):
            change_t = F.interpolate(change_t, size=(img_height, img_width),
                                     mode="bilinear", align_corners=False)
        if q1_t.shape[-2:] != (img_height, img_width):
            q1_t = F.interpolate(q1_t, size=(img_height, img_width),
                                 mode="bilinear", align_corners=False)
        if q2_t.shape[-2:] != (img_height, img_width):
            q2_t = F.interpolate(q2_t, size=(img_height, img_width),
                                 mode="bilinear", align_corners=False)

        change_np = change_t[0, 0].detach().cpu().numpy()
        q1_np = F.softmax(q1_t, dim=1)[0].detach().cpu().numpy()
        q2_np = F.softmax(q2_t, dim=1)[0].detach().cpu().numpy()
        gt_np = to_numpy_label(batch["gt"])
        l1_np = to_numpy_label(batch["label1"])
        l2_np = to_numpy_label(batch["label2"])

        for name, arr in (("gt", gt_np), ("label1", l1_np), ("label2", l2_np)):
            if arr.shape != (img_height, img_width):
                raise ValueError(
                    f"{name} shape {arr.shape} does not match full image "
                    f"{(img_height, img_width)}."
                )

        canvas_change[:, :] = change_np
        canvas_gt_change[:, :] = gt_np
        canvas_q1[:, :, :] = q1_np
        canvas_q2[:, :, :] = q2_np
        canvas_l1[:, :] = l1_np
        canvas_l2[:, :] = l2_np

    s_mean = s_sum / max(n_batches, 1)
    d_mean = d_sum / max(n_batches, 1)
    unsup_score = s_mean - d_mean

    cm_save, otsu_thr = compute_otsu_binary(canvas_change)

    sem1_raw = canvas_q1.argmax(axis=0).astype(np.uint8)
    sem2_raw = canvas_q2.argmax(axis=0).astype(np.uint8)

    gt_binary, binary_valid = make_valid_binary_gt(canvas_gt_change)
    binary_metrics = evaluate_binary_saved(
        cm_save, gt_binary, valid_mask=binary_valid
    )

    all_pixels = np.ones((img_height, img_width), dtype=bool)
    sem1_all = evaluate_semantic_saved(sem1_raw, canvas_l1, all_pixels, K)
    sem2_all = evaluate_semantic_saved(sem2_raw, canvas_l2, all_pixels, K)

    scd_metrics = scd_metrics_saved(
        sem1_raw, sem2_raw,
        canvas_l1, canvas_l2,
        gt_binary, K
    )

    metrics = {
        "binary": binary_metrics,
        "sem1_all": sem1_all,
        "sem2_all": sem2_all,
        "scd": scd_metrics,
        "otsu_thr": float(otsu_thr),
        "s_mean": float(s_mean),
        "d_mean": float(d_mean),
        "score": float(unsup_score)
    }

    arrays = {
        "change": canvas_change,
        "gt_change": gt_binary,
        "cm": cm_save,
        "sem1_raw": sem1_raw,
        "sem2_raw": sem2_raw,
        "label1": canvas_l1,
        "label2": canvas_l2
    }
    return metrics, arrays



CLASS_COLORS_BGR = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
    4: (0, 255, 255)
}


def colorize_label(label):
    label = np.asarray(label)
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls, bgr in CLASS_COLORS_BGR.items():
        color[label == cls] = bgr
    for cls in np.unique(label):
        cls = int(cls)
        if cls not in CLASS_COLORS_BGR:
            rng = np.random.default_rng(cls + 123)
            color[label == cls] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return color


def save_semantic_metrics(f, name, m):
    f.write(f"\n[{name}]\n")
    if m is None:
        f.write("No valid pixels.\n")
        return

    for key in ["OA", "Hungarian_ACC", "ARI", "NMI",
                "mean_F1", "mIoU", "n"]:
        if key in m:
            f.write(f"{key} = {m[key]}\n")

    f.write(f"classes = {m.get('classes', [])}\n")
    for cls, values in m.get("per_class", {}).items():
        f.write(f"class {cls}: {values}\n")


def save_results(arrays, metrics, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    change = np.clip(arrays["change"], 0, 1)

    io.imsave(
        os.path.join(output_dir, "DI.png"),
        np.round(change * 255).astype(np.uint8),
        check_contrast=False
    )

    io.imsave(
        os.path.join(output_dir, "CM.png"),
        (arrays["cm"].astype(np.uint8) * 255),
        check_contrast=False
    )

    io.imsave(
        os.path.join(output_dir, "semantic_T1.png"),
        arrays["sem1_raw"].astype(np.uint8),
        check_contrast=False
    )
    io.imsave(
        os.path.join(output_dir, "semantic_T2.png"),
        arrays["sem2_raw"].astype(np.uint8),
        check_contrast=False
    )

    cv2.imwrite(
        os.path.join(output_dir, "semantic_T1_color.png"),
        colorize_label(arrays["sem1_raw"])
    )
    cv2.imwrite(
        os.path.join(output_dir, "semantic_T2_color.png"),
        colorize_label(arrays["sem2_raw"])
    )

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes[0, 0].imshow(arrays["change"], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("DI probability")
    axes[0, 1].imshow(arrays["cm"], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"CM (Otsu={metrics['otsu_thr']:.3f})")
    axes[0, 2].imshow(arrays["gt_change"], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Binary GT")
    axes[0, 3].imshow(arrays["label1"], cmap="tab10")
    axes[0, 3].set_title("Semantic GT T1")

    axes[1, 0].imshow(arrays["sem1_raw"], cmap="tab10", vmin=0, vmax=args.K - 1)
    axes[1, 0].set_title("T1 semantic")
    axes[1, 1].imshow(arrays["sem2_raw"], cmap="tab10", vmin=0, vmax=args.K - 1)
    axes[1, 1].set_title("T2 semantic")
    axes[1, 2].imshow(arrays["label1"], cmap="tab10", vmin=0, vmax=args.K - 1)
    axes[1, 2].set_title("GT T1")
    axes[1, 3].imshow(arrays["label2"], cmap="tab10", vmin=0, vmax=args.K - 1)
    axes[1, 3].set_title("GT T2")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "visualization.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    with open(os.path.join(output_dir, "eval.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset = {dataset_name}\n")
        f.write(f"Otsu threshold = {metrics['otsu_thr']:.6f}\n")
        f.write("Training labels: pseudo, pseudo_l1, pseudo_l2 only.\n")
        f.write("GT labels are used only for evaluation and BCD-F1 checkpoint selection.\n")
        f.write("Semantic maps are RAW, without CM masking.\n")
        f.write("Evaluation protocol: every pixel in the full image is included; class 0 is not filtered.\n")
        f.write("Binary GT: 0=unchanged, positive=changed.\n")
        f.write("SCD semantic change diagnostic: gt_t1 != gt_t2, evaluated on all pixels.\n")

        f.write("\n[Unsupervised score]\n")
        f.write(f"s_mean = {metrics['s_mean']:.6f}\n")
        f.write(f"d_mean = {metrics['d_mean']:.6f}\n")
        f.write(f"score = {metrics['score']:.6f}\n")

        b = metrics["binary"]
        f.write("\n[Binary Change Detection]\n")
        if b is None:
            f.write("No valid binary pixels.\n")
        else:
            for key in ["OA", "Precision", "Recall", "F1", "IoU",
                        "Kappa", "Specificity", "TP", "TN", "FP", "FN",
                        "n_valid"]:
                f.write(f"{key} = {b[key]}\n")

        save_semantic_metrics(f, "Semantic T1, ALL image pixels including class 0",
                              metrics["sem1_all"])
        save_semantic_metrics(f, "Semantic T2, ALL image pixels including class 0",
                              metrics["sem2_all"])

        f.write("\n[SCD]\n")
        scd = metrics["scd"]
        if scd is None:
            f.write("No valid SCD pixels.\n")
        else:
            for key in [
                "OA", "Pre", "Rec", "Fscd", "mIoU", "Sek", "Score",
                "Pre_scd", "Rec_scd", "n_valid", "gt_classes",
                "gt_change_mismatch", "TP_cd", "FP_cd", "FN_cd", "mapping"
            ]:
                f.write(f"{key} = {scd.get(key)}\n")
            for cls, values in scd["per_class"].items():
                f.write(f"class {cls}: {values}\n")


def main():
    output_root = args.result_folder
    result_dir = os.path.join(output_root, "results")
    os.makedirs(result_dir, exist_ok=True)

    log_path = os.path.join(output_root, "train_log.txt")
    best_f1 = -float("inf")
    best_epoch = -1

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Epochs: {args.epoch}\n")
        f.write(f"K: {args.K}\n")
        f.write("Best checkpoint selected by BCD F1 on the evaluation set.\n")
        f.write("=" * 90 + "\n")

    for epoch in range(args.epoch):
        avg_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch
        )
        scheduler.step()

        metrics, arrays = evaluate_epoch(test_loader, model)
        b = metrics["binary"]
        s = metrics["scd"]

        print(f"\n[Epoch {epoch + 1}] loss={avg_loss:.4f}")

        if b is not None:
            print(
                f"  [BCD] F1={b['F1']:.4f} IoU={b['IoU']:.4f} "
                f"OA={b['OA']:.4f} Kappa={b['Kappa']:.4f} "
                f"(TP={b['TP']} TN={b['TN']} FP={b['FP']} FN={b['FN']})"
            )

        for name in ["sem1_all", "sem2_all"]:
            sm = metrics[name]
            if sm is not None:
                print(
                    f"  [{name}] OA={sm['OA']:.4f} "
                    f"mIoU={sm['mIoU']:.4f} "
                    f"mF1={sm['mean_F1']:.4f} n={sm['n']}"
                )

        if s is not None:
            print(
                f"  [SCD] OA={s['OA']:.4f} Pre={s['Pre']:.4f} "
                f"Rec={s['Rec']:.4f} Fscd={s['Fscd']:.4f} "
                f"mIoU={s['mIoU']:.4f} Sek={s['Sek']:.4f} "
                f"Score={s['Score']:.4f} n_valid={s['n_valid']}"
            )

        # Append epoch log.
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch + 1:03d} | loss={avg_loss:.6f} "
                f"| unsup_score={metrics['score']:.6f} "
                f"| s_mean={metrics['s_mean']:.6f} "
                f"| d_mean={metrics['d_mean']:.6f}"
            )
            if b is not None:
                f.write(
                    f" | BCD_F1={b['F1']:.6f}"
                    f" | BCD_IoU={b['IoU']:.6f}"
                    f" | BCD_OA={b['OA']:.6f}"
                )
            if s is not None:
                f.write(
                    f" | SCD_OA={s['OA']:.6f}"
                    f" | SCD_Pre={s['Pre']:.6f}"
                    f" | SCD_Rec={s['Rec']:.6f}"
                    f" | SCD_Fscd={s['Fscd']:.6f}"
                    f" | SCD_mIoU={s['mIoU']:.6f}"
                    f" | SCD_Sek={s['Sek']:.6f}"
                    f" | SCD_Score={s['Score']:.6f}"
                    f" | GT_change_mismatch={s['gt_change_mismatch']:.6f}"
                )
            f.write("\n")

        if b is not None and np.isfinite(b['F1']) and b['F1'] > best_f1:
            best_f1 = b['F1']
            best_epoch = epoch + 1

            print(
                f"\n[BEST UPDATE] epoch={best_epoch} "
                f"BCD_F1={best_f1:.4f}\n"
            )

            # Clear previous result files.
            for name in os.listdir(result_dir):
                path = os.path.join(result_dir, name)
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "metrics": metrics,
                    "best_scd_f1": best_f1,
                    "args": vars(args)
                },
                os.path.join(result_dir, "best_model.pth")
            )

            save_results(arrays, metrics, result_dir)

            with open(
                    os.path.join(result_dir, "best_epoch.txt"),
                    "w", encoding="utf-8"
            ) as f:
                f.write(f"Dataset = {dataset_name}\n")
                f.write(f"Best epoch = {best_epoch}\n")
                f.write(f"Best SCD F1 = {best_f1:.8f}\n")
                f.write("Checkpoint selected by BCD F1 on the evaluation set.\n")

    print("\nTraining completed.")
    print(f"Best epoch = {best_epoch}")
    print(f"Best SCD F1 = {best_f1:.6f}")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()

