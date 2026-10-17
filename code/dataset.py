# dataset.py
import os
import cv2
import glob
import random
import numpy as np
import torch
import torch.utils.data


dataset_1  = r'/home/user/storage/mainBack/zly/datasets/shuguang'
dataset_2  = r'/home/user/storage/mainBack/zly/datasets/Italy'
dataset_3  = r'/home/user/storage/mainBack/zly/datasets/yellow_C'
dataset_4  = r'/home/user/storage/mainBack/zly/datasets/yellow_D'
dataset_5  = r'/home/user/storage/mainBack/zly/datasets/Img17'


datasets = [
    dataset_1, dataset_2, dataset_3, dataset_4, dataset_5,
]


MODE_MAP = {
    'shuguang':  ('sar', 'optical'),
    'Italy':     ('sar', 'optical'),
    'yellow_C':  ('sar', 'sar'),
    'yellow_D':  ('sar', 'sar'),
    'Img17':     ('sar', 'optical'),

}


# ============================================================
# 类别均值（仅作为伪类别缺失时的回退）
# 类别编号：1=水体, 2=建筑, 3=植被, 4=土壤
# ============================================================
OPTICAL_MEANS = {
    1: [44.51, 42.86, 26.59],
    2: [192.11, 160.02, 147.16],
    3: [111.16, 82.62, 149.65],
    4: [124.18, 129.71, 138.39],
}

SAR_MEANS = {
    1: [54.27],
    2: [190.16],
    3: [113.41],
    4: [145.40],
}


def normalize_image(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _find_one(dataset, patterns):
    for p in patterns:
        files = glob.glob(os.path.join(dataset, p))
        if files:
            return files[0]
    return None


def find_dataset_files(dataset):
    name = os.path.basename(dataset)
    mode_1, mode_2 = MODE_MAP.get(name, ('sar', 'optical'))

    if mode_1 == 'sar' and mode_2 == 'sar':

        pat_1 = ['*_1.*', '*_t1.*', '*_T1.*', '*_before.*',
                 '*_A.*', '*_a.*', '*_sar_clip.*', '*_SAR.*']
        pat_2 = ['*_2.*', '*_t2.*', '*_T2.*', '*_after.*',
                 '*_B.*', '*_b.*']

        Im_sar = _find_one(dataset, pat_1)
        Im_optical = _find_one(dataset, pat_2)

        # 兜底：模式没命中，或命中同一个文件 -> 用所有非 gt/label 文件排序取前两个
        if Im_sar is None or Im_optical is None or Im_sar == Im_optical:
            all_files = sorted(
                f for f in glob.glob(os.path.join(dataset, '*'))
                if os.path.isfile(f) and not _is_gt_or_label(f)
            )
            if len(all_files) >= 2:
                Im_sar, Im_optical = all_files[0], all_files[1]
            else:
                raise FileNotFoundError(
                    f"SAR-SAR 数据不足两张: {dataset}, files={all_files}"
                )
    else:

        if mode_1 == 'sar':
            pat_1 = ['*_1.*', '*_sar_clip.*', '*_SAR.*', '*_sar.*']
        else:
            pat_1 = ['*_2.*', '*_optial_clip.*', '*_OPTIAL.*', '*_optical.*']

        if mode_2 == 'sar':
            pat_2 = ['*_2.*', '*_sar_clip.*', '*_SAR.*', '*_sar.*']
        else:
            pat_2 = ['*_2.*', '*_optial_clip.*', '*_OPTIAL.*', '*_optical.*']

        Im_sar = _find_one(dataset, pat_1)
        Im_optical = _find_one(dataset, pat_2)

        if Im_sar is not None and Im_sar == Im_optical:
            all_files = sorted(
                f for f in glob.glob(os.path.join(dataset, '*'))
                if os.path.isfile(f) and not _is_gt_or_label(f)
            )
            if len(all_files) >= 2:
                Im_sar, Im_optical = all_files[0], all_files[1]

    gt_path = _find_one(dataset, ['*_gt.*', '*_GT.*', '*_label.*'])
    label1_path = _find_one(dataset, ['*_label1.*', '*_label_1.*'])
    label2_path = _find_one(dataset, ['*_label2.*', '*_label_2.*'])

    if Im_sar is None or Im_optical is None:
        raise FileNotFoundError(f"两期影像未找齐: {dataset}")
    if gt_path is None:
        raise FileNotFoundError(f"GT 未找到: {dataset}")
    if label1_path is None or label2_path is None:
        raise FileNotFoundError(f"label1/label2 未找到: {dataset}")
    return Im_sar, Im_optical, gt_path, label1_path, label2_path


def _is_gt_or_label(path):
    base = os.path.basename(path).lower()
    return any(k in base for k in ['gt', 'label'])

def find_pseudo_label(dataset, pseudo_name=None):

    if pseudo_name is not None:
        candidates = [
            os.path.join(dataset, pseudo_name),
            os.path.join(dataset, 'Pseudo', pseudo_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    cand_dirs = [
        os.path.join(dataset, 'Pseudo'),
        os.path.join(dataset, 'pseudo_USSD'),
    ]
    patterns = ['CM_map_iter_6.png', 'USSD.tif', '*pseudo*.png', '*pseudo*.tif']
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            files = glob.glob(os.path.join(d, p))
            if files:
                return files[0]
    return None


def find_pseudo_semantic(dataset, name):

    cand_dirs = [
        os.path.join(dataset, 'Pseudo'),
        os.path.join(dataset, 'pseudo_sem'),
        dataset,
    ]
    patterns = [
        f'{name}.png', f'{name}.tif',
        f'*{name}*.png', f'*{name}*.tif',
        f'*{name}*.tiff',
    ]
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            files = glob.glob(os.path.join(d, p))
            if files:
                return files[0]
    return None


def _read_image(path, color=True):
    if color:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"读图失败: {path}")
    return img


def _load_label(path):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"label 读取失败: {path}")

    if img.ndim == 2:
        return img.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 3:
        img = img.astype(np.uint8)
        c0, c1, c2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if np.array_equal(c0, c1) and np.array_equal(c0, c2):
            return c0.astype(np.uint8)

        # BGR 真彩色 -> 0~4
        label = np.zeros(img.shape[:2], dtype=np.uint8)
        color_map = {
            (0, 0, 0):       0,   # 黑 -> 背景
            (255, 0, 0):     1,   # 蓝 -> 水体
            (0, 0, 255):     2,   # 红 -> 建筑
            (0, 255, 0):     3,   # 绿 -> 植被
            (0, 255, 255):   4,   # 黄 -> 土壤
        }
        for color, cls in color_map.items():
            mask = np.all(img == np.array(color, dtype=np.uint8), axis=-1)
            label[mask] = cls
        return label

    raise RuntimeError(f"无法解析 label: {path}, shape={img.shape}")


def coarse_classify(img, means, mode='rgb', change_mask=None):

    H, W = img.shape[:2]
    classes = sorted(means.keys())
    dists = []
    for c in classes:
        m = np.array(means[c], dtype=np.float32)
        if mode == 'rgb':
            d = np.linalg.norm(img - m[None, None, :], axis=-1)
        else:
            d = np.abs(img[:, :, 0] - m[0])
        dists.append(d)
    dists = np.stack(dists, axis=0)
    idx = np.argmin(dists, axis=0)
    label = (idx + 1).astype(np.uint8)

    if change_mask is not None:
        label = label * change_mask.astype(np.uint8)
    return label


def _read_pair_and_labels(dataset, pseudo_name=None):
    Im_sar_path, Im_optical_path, gt_path, label1_path, label2_path = \
        find_dataset_files(dataset)

    Im_sar = _read_image(Im_sar_path, color=True)
    Im_optical = _read_image(Im_optical_path, color=True)
    gt = _read_image(gt_path, color=False)
    label1 = _load_label(label1_path)
    label2 = _load_label(label2_path)

    if Im_sar.ndim == 2:
        Im_sar = np.stack([Im_sar] * 3, axis=-1)
    if Im_optical.ndim == 2:
        Im_optical = np.stack([Im_optical] * 3, axis=-1)

    Im_sar = normalize_image(Im_sar)
    Im_optical = normalize_image(Im_optical)
    gt = (gt > 127).astype(np.uint8)


    pseudo_path = find_pseudo_label(dataset, pseudo_name)
    if pseudo_path is not None:
        pseudo = cv2.imread(pseudo_path, cv2.IMREAD_GRAYSCALE)
        if pseudo is None:
            raise RuntimeError(f"pseudo 读取失败: {pseudo_path}")
        pseudo = (pseudo > 127).astype(np.uint8)
        print(f"[dataset] pseudo: {pseudo_path}")
    else:
        diff = np.abs(Im_sar - Im_optical).mean(axis=2)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        thr = np.quantile(diff, 0.9)
        pseudo = (diff > thr).astype(np.uint8)
        print(f"[dataset] 未找到 pseudo，CVA 生成 (thr={thr:.3f})")

    H, W = Im_sar.shape[:2]
    if Im_optical.shape[:2] != (H, W):
        Im_optical = cv2.resize(Im_optical, (W, H), interpolation=cv2.INTER_LINEAR)
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    if label1.shape != (H, W):
        label1 = cv2.resize(label1, (W, H), interpolation=cv2.INTER_NEAREST)
    if label2.shape != (H, W):
        label2 = cv2.resize(label2, (W, H), interpolation=cv2.INTER_NEAREST)
    if pseudo.shape != (H, W):
        pseudo = cv2.resize(pseudo, (W, H), interpolation=cv2.INTER_NEAREST)


    pseudo_l1_path = find_pseudo_semantic(dataset, 'pseudo_label1')
    pseudo_l2_path = find_pseudo_semantic(dataset, 'pseudo_label2')

    name = os.path.basename(dataset)
    mode_sar, mode_opt = MODE_MAP.get(name, ('sar', 'optical'))

    if pseudo_l1_path is not None:
        pseudo_l1 = _load_label(pseudo_l1_path)
        if pseudo_l1.shape != (H, W):
            pseudo_l1 = cv2.resize(pseudo_l1, (W, H),
                                   interpolation=cv2.INTER_NEAREST)
        print(f"[dataset] pseudo_label1: {pseudo_l1_path}")
    else:
        Im_sar_255 = (Im_sar * 255.0).astype(np.float32)
        if mode_sar == 'sar':
            pseudo_l1 = coarse_classify(Im_sar_255, SAR_MEANS,
                                        mode='gray', change_mask=pseudo)
        else:
            pseudo_l1 = coarse_classify(Im_sar_255, OPTICAL_MEANS,
                                        mode='rgb', change_mask=pseudo)
        print(f"[dataset] 未找到 pseudo_label1，coarse 生成")

    if pseudo_l2_path is not None:
        pseudo_l2 = _load_label(pseudo_l2_path)
        if pseudo_l2.shape != (H, W):
            pseudo_l2 = cv2.resize(pseudo_l2, (W, H),
                                   interpolation=cv2.INTER_NEAREST)
        print(f"[dataset] pseudo_label2: {pseudo_l2_path}")
    else:
        Im_opt_255 = (Im_optical * 255.0).astype(np.float32)
        if mode_opt == 'sar':
            pseudo_l2 = coarse_classify(Im_opt_255, SAR_MEANS,
                                        mode='gray', change_mask=pseudo)
        else:
            pseudo_l2 = coarse_classify(Im_opt_255, OPTICAL_MEANS,
                                        mode='rgb', change_mask=pseudo)
        print(f"[dataset] 未找到 pseudo_label2，coarse 生成")

    # ---- 非变化区域强制置 0 ----
    pseudo_l1 = (pseudo_l1 * pseudo).astype(np.uint8)
    pseudo_l2 = (pseudo_l2 * pseudo).astype(np.uint8)

    return (Im_sar, Im_optical, gt, label1, label2, pseudo,
            pseudo_l1, pseudo_l2)


class OpticalSARChangeDataset(torch.utils.data.Dataset):

    def __init__(self, train=True, dataset_id=1,
                 transform=None, repeat=1,
                 augment=False, pseudo_name=None):
        super().__init__()
        dataset = datasets[dataset_id - 1]
        self.train = train
        self.transform = transform
        self.repeat = repeat if train else 1
        self.augment = augment
        self.dataset_name = os.path.basename(dataset)

        (Im_sar, Im_optical, gt, label1, label2, pseudo,
         pseudo_l1, pseudo_l2) = _read_pair_and_labels(dataset, pseudo_name)

        self.H, self.W = Im_sar.shape[:2]

        self.Im_sar = torch.from_numpy(Im_sar.transpose(2, 0, 1)).float()
        self.Im_optical = torch.from_numpy(Im_optical.transpose(2, 0, 1)).float()
        self.gt = gt
        self.label1 = label1
        self.label2 = label2
        self.pseudo = pseudo
        self.pseudo_l1 = pseudo_l1
        self.pseudo_l2 = pseudo_l2

    def __len__(self):
        return self.repeat

    def __getitem__(self, index):
        Im_sar = self.Im_sar.clone()
        Im_optical = self.Im_optical.clone()
        pseudo_l1 = torch.from_numpy(self.pseudo_l1.astype(np.int64))
        pseudo_l2 = torch.from_numpy(self.pseudo_l2.astype(np.int64))

        if self.train:
            pseudo = torch.from_numpy(
                self.pseudo.astype(np.float32)
            ).unsqueeze(0)

            if self.augment and self.transform is not None:
                seed = random.randint(0, 2 ** 31 - 1)
                random.seed(seed); np.random.seed(seed)
                Im_sar_np = self.transform(Im_sar.permute(1, 2, 0).numpy())
                random.seed(seed); np.random.seed(seed)
                Im_opt_np = self.transform(Im_optical.permute(1, 2, 0).numpy())
                Im_sar = torch.from_numpy(Im_sar_np.transpose(2, 0, 1)).float()
                Im_optical = torch.from_numpy(Im_opt_np.transpose(2, 0, 1)).float()

            return {
                'Im_sar': Im_sar,
                'Im_optical': Im_optical,
                'pseudo': pseudo,
                'pseudo_l1': pseudo_l1,
                'pseudo_l2': pseudo_l2,
            }
        else:
            gt = torch.from_numpy(self.gt.astype(np.float32)).unsqueeze(0)
            label1 = torch.from_numpy(self.label1.astype(np.int64))
            label2 = torch.from_numpy(self.label2.astype(np.int64))
            pseudo = torch.from_numpy(self.pseudo.astype(np.float32)).unsqueeze(0)

            return {
                'Im_sar': Im_sar,
                'Im_optical': Im_optical,
                'gt': gt,
                'label1': label1,
                'label2': label2,
                'pseudo': pseudo,
                'pseudo_l1': pseudo_l1,
                'pseudo_l2': pseudo_l2,
            }


if __name__ == "__main__":

    for did in [1, 2, 3, 4, 5, 6]:
        try:
            ds = OpticalSARChangeDataset(train=True, dataset_id=did)
            s = ds[0]
            print(f"[dataset_id={did}] {ds.dataset_name}")
            print("  Im_sar     :", s['Im_sar'].shape)
            print("  Im_optical :", s['Im_optical'].shape)
            print("  pseudo     :", s['pseudo'].shape,
                  "unique:", torch.unique(s['pseudo']).tolist())
            print("  pseudo_l1  :", s['pseudo_l1'].shape,
                  "unique:", torch.unique(s['pseudo_l1']).tolist())
            print("  pseudo_l2  :", s['pseudo_l2'].shape,
                  "unique:", torch.unique(s['pseudo_l2']).tolist())
        except Exception as e:
            print(f"[dataset_id={did}] ERROR: {e}")

    ds = OpticalSARChangeDataset(train=False, dataset_id=5)
    s = ds[0]
    print("\n[test] Img17")
    print("  gt         :", s['gt'].shape,
          "unique:", torch.unique(s['gt']).tolist())
    print("  label1     :", s['label1'].shape,
          "unique:", torch.unique(s['label1']).tolist())
    print("  label2     :", s['label2'].shape,
          "unique:", torch.unique(s['label2']).tolist())