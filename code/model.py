# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c):
    g = 8
    while c % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, c)


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps

        if affine:
            self.gamma = nn.Parameter(
                torch.ones(1, num_channels, 1, 1)
            )
            self.beta = nn.Parameter(
                torch.zeros(1, num_channels, 1, 1)
            )
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.gamma is not None:
            y = y * self.gamma + self.beta
        return y

LocalContrastNorm = LayerNorm2d


def local_relation_descriptor(x, mode='optical', ks=5, tau=1.0, eps=1e-6):

    B, C, H, W = x.shape
    pad = ks // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=ks)
    patches = patches.view(B, C, ks * ks, H, W)

    if mode == 'optical':

        mean = patches.mean(dim=2, keepdim=True)
        std  = patches.std(dim=2, keepdim=True) + eps
        p_n  = (patches - mean) / std
        c_n  = (x - mean.squeeze(2)) / std.squeeze(2)
        diff = (c_n.unsqueeze(2) - p_n).abs().mean(dim=1)
    else:

        center = torch.log(x + eps)
        neigh  = torch.log(patches + eps)
        diff = torch.abs(center.unsqueeze(2) - neigh).mean(dim=1)

    scale = diff.mean(dim=1, keepdim=True) + eps
    diff_norm = diff / scale
    A = torch.exp(-diff_norm / tau)

    desc = torch.cat([
        diff_norm,
        diff_norm.mean(dim=1, keepdim=True),
        diff_norm.std(dim=1, keepdim=True),
        A.mean(dim=1, keepdim=True),
        A.max(dim=1, keepdim=True)[0],
    ], dim=1)
    return desc, diff_norm


class RelationalEncoder(nn.Module):
    def __init__(self, in_ch, emb_dim=64):
        super().__init__()
        self.in_norm = LayerNorm2d(in_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            _gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, emb_dim, 3, padding=2, dilation=2, bias=False),
            _gn(emb_dim), nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim, emb_dim, 1),
        )

    def forward(self, x):
        return F.normalize(self.net(self.in_norm(x)), dim=1)


class SemanticHead(nn.Module):
    def __init__(self, emb_dim=64, num_classes=5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(emb_dim, 128, 3, padding=1, bias=False),
            _gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, f):
        return self.proj(f)


class ChangeHead(nn.Module):
    def __init__(self, emb_dim=64, num_classes=5):
        super().__init__()

        in_ch = emb_dim * 3 + num_classes * 2 + 1 + 1
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            _gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2, bias=False),
            _gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, z_O, z_S, q1, q2, s01, r_cross):
        diff = torch.abs(z_O - z_S)
        q1_soft = F.softmax(q1, dim=1)
        q2_soft = F.softmax(q2, dim=1)
        q_diff = torch.abs(q1_soft - q2_soft)
        q_prod = q1_soft * q2_soft
        x = torch.cat([diff, z_O, z_S, q_diff, q_prod, s01, r_cross], dim=1)
        assert x.size(1) == self.in_ch, \
            f'ChangeHead in_ch mismatch: {x.size(1)} vs {self.in_ch}'
        return self.net(x)


class SemanticTransitionHead(nn.Module):


    def __init__(self, num_classes=5, hidden=32):
        super().__init__()
        self.num_classes = num_classes
        in_ch = num_classes * num_classes

        self.net = nn.Sequential(

            nn.Conv2d(in_ch, hidden, 1, bias=False),
            _gn(hidden), nn.ReLU(inplace=True),


            nn.Conv2d(hidden, hidden, 3,
                      padding=1, groups=hidden, bias=False),
            _gn(hidden), nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 1, bias=False),
            _gn(hidden), nn.ReLU(inplace=True),


            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, q1, q2):
        if q1.shape != q2.shape:
            raise ValueError("q1 和 q2 的形状必须一致")

        B, K, H, W = q1.shape
        if K != self.num_classes:
            raise ValueError(
                f"类别数错误: {K} != {self.num_classes}"
            )

        transition = torch.einsum('bihw,bjhw->bijhw', q1, q2)

        transition_features = transition.reshape(B, K * K, H, W)

        transition_logits = self.net(transition_features)
        return transition_logits, transition


class CRST_CD(nn.Module):
    def __init__(
        self,
        in_channels=3,
        emb_dim=64,
        ks=5,
        num_classes=5,
        transition_hidden=32,
        transition_init_scale=0.1,
    ):
        super().__init__()
        self.ks = ks
        self.num_classes = num_classes
        desc_ch = ks * ks + 4
        self.enc_O = RelationalEncoder(desc_ch, emb_dim)
        self.enc_S = RelationalEncoder(desc_ch, emb_dim)
        self.sem_head = SemanticHead(emb_dim, num_classes)
        self.change_head = ChangeHead(emb_dim, num_classes)

        self.transition_head = SemanticTransitionHead(
            num_classes=num_classes,
            hidden=transition_hidden,
        )

        self.transition_scale = nn.Parameter(
            torch.tensor(float(transition_init_scale))
        )

    def forward(self, O, S):

        R_O, r_diff_O = local_relation_descriptor(O, 'optical', ks=self.ks)
        R_S, r_diff_S = local_relation_descriptor(S, 'sar', ks=self.ks)

        z_O = self.enc_O(R_O)
        z_S = self.enc_S(R_S)

        q1_logits = self.sem_head(z_O)
        q2_logits = self.sem_head(z_S)
        q1_soft = F.softmax(q1_logits, dim=1)
        q2_soft = F.softmax(q2_logits, dim=1)

        r_cross = (r_diff_O - r_diff_S).abs().mean(dim=1, keepdim=True)

        s = (z_O * z_S).sum(dim=1, keepdim=True).clamp(-1, 1)
        s01 = (s + 1.0) / 2.0
        d_cross = 1.0 - s01

        base_change_logits = self.change_head(
            z_O, z_S, q1_logits, q2_logits, s01, r_cross)

        transition_logits, transition = self.transition_head(
            q1_soft, q2_soft)

        change_logits = (
            base_change_logits
            + self.transition_scale * transition_logits
        )
        change_prob = torch.sigmoid(change_logits)

        bg_bias = (1.0 - change_prob) * 10.0
        q1_logits_gated = q1_logits.clone()
        q2_logits_gated = q2_logits.clone()
        q1_logits_gated[:, 0:1] = q1_logits_gated[:, 0:1] + bg_bias
        q2_logits_gated[:, 0:1] = q2_logits_gated[:, 0:1] + bg_bias

        return {
            'change_logits': change_logits,
            'change_prob': change_prob,

            'base_change_logits': base_change_logits,
            'transition_logits': transition_logits,
            'transition_scale': self.transition_scale,

            'q1_logits': q1_logits,
            'q2_logits': q2_logits,
            'q1_logits_gated': q1_logits_gated,
            'q2_logits_gated': q2_logits_gated,
            'q1': q1_soft,
            'q2': q2_soft,

            's': s01,
            'd_cross': d_cross,
            'r_cross': r_cross,

            'transition': transition,

            'z_O': z_O,
            'z_S': z_S,
        }



class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logit, target):
        if logit.dim() == 4 and logit.size(1) == 1:
            logit = logit[:, 0]
        if target.dim() == 4:
            target = target[:, 0]
        prob = torch.sigmoid(logit).reshape(logit.size(0), -1)
        tgt = target.reshape(target.size(0), -1).float()
        inter = (prob * tgt).sum(dim=1)
        union = prob.sum(dim=1) + tgt.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class CRST_CD_Loss(nn.Module):

    def __init__(
        self,
        w_change=1.0,
        w_sem_change=2.0,
        w_change_sem=0.1,
        num_classes=5,
        ignore_index=255,
        class_weight=None,
    ):
        super().__init__()
        self.w_change = w_change
        self.w_sem_change = w_sem_change
        self.w_change_sem = w_change_sem
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if class_weight is None:
            class_weight = [0.1] + [1.0] * (num_classes - 1)
        self.register_buffer(
            "class_weight",
            torch.tensor(class_weight, dtype=torch.float32)
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    @staticmethod
    def _ensure_4d(p):
        return p.unsqueeze(1) if p.dim() == 3 else p

    @staticmethod
    def _ensure_3d_label(p):
        if p.dim() == 4:
            assert p.size(1) == 1, f"label C={p.size(1)} != 1"
            p = p[:, 0]
        return p.long()

    def change_loss(self, logits, pseudo):
        pseudo = self._ensure_4d(pseudo).float()
        return self.bce(logits, pseudo) + self.dice(logits, pseudo)

    def sem_loss(self, q_logits, pseudo_sem, pseudo_change=None,
                 use_change_as_valid=True):

        target = self._ensure_3d_label(pseudo_sem)

        valid = (target != self.ignore_index)
        if not use_change_as_valid and pseudo_change is not None:
            change_bin = (self._ensure_4d(pseudo_change)[:, 0] > 0.5)
            valid = valid & (~change_bin)

        if valid.sum() == 0:
            return q_logits.new_zeros(())

        loss_map = F.cross_entropy(
            q_logits, target,
            weight=self.class_weight.to(q_logits.device),
            reduction='none',
            ignore_index=self.ignore_index,
        )
        loss = (loss_map * valid.float()).sum() / (valid.float().sum() + 1e-6)
        return loss

    def change_region_sem_loss(self, q_logits, pseudo_sem, pseudo_change):

        change_bin = (self._ensure_4d(pseudo_change)[:, 0] > 0.5)
        target = self._ensure_3d_label(pseudo_sem)

        mask = change_bin & (target > 0) & (target != self.ignore_index)
        if mask.sum() == 0:
            return q_logits.new_zeros(())

        loss_map = F.cross_entropy(
            q_logits, target,
            reduction='none',
            ignore_index=self.ignore_index,
        )
        return (loss_map * mask.float()).sum() / (mask.float().sum() + 1e-6)

    def spatial_loss(self, q):
        q_pad = F.pad(q, (1, 1, 1, 1), mode='reflect')
        neighs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighs.append(
                    q_pad[:, :, 1+dx:1+dx+q.shape[2],
                            1+dy:1+dy+q.shape[3]])
        neighs = torch.stack(neighs, dim=2)
        return (q.unsqueeze(2) - neighs).abs().mean()

    def _js(self, q1, q2, eps=1e-8):
        q1_c = q1.clamp(min=eps)
        q2_c = q2.clamp(min=eps)
        m = 0.5 * (q1_c + q2_c)
        js = 0.5 * (q1_c * (q1_c / m).log()).sum(1, keepdim=True) + \
             0.5 * (q2_c * (q2_c / m).log()).sum(1, keepdim=True)
        return js

    def change_sem_loss(self, q1, q2, pseudo, margin=0.2):

        change_mask = self._ensure_4d(pseudo).float()
        js = self._js(q1, q2)
        loss = F.relu(margin - js) * change_mask
        return loss.sum() / (change_mask.sum() + 1e-6)

    def forward(self, out, pseudo, pseudo_l1=None, pseudo_l2=None):
        loss = self.w_change * self.change_loss(out['change_logits'], pseudo)

        if pseudo_l1 is not None:

            loss = loss + self.w_sem_change * self.change_region_sem_loss(
                out['q1_logits'], pseudo_l1, pseudo)
        if pseudo_l2 is not None:

            loss = loss + self.w_sem_change * self.change_region_sem_loss(
                out['q2_logits'], pseudo_l2, pseudo)

        loss = loss + self.w_change_sem * self.change_sem_loss(
            out['q1'], out['q2'], pseudo)

        return loss

def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    x = x.astype(np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return 0.5
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    hist, edges = np.histogram(x_norm, bins=nbins, range=(0, 1))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b))
    return float(centers[k])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRST_CD(in_channels=3, emb_dim=32, ks=5, num_classes=5).to(device)
    criterion = CRST_CD_Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    O = torch.randn(2, 3, 256, 256).to(device)
    S = torch.randn(2, 3, 256, 256).to(device)
    pseudo = (torch.rand(2, 1, 256, 256) > 0.7).float().to(device)
    pseudo_l1 = torch.randint(0, 5, (2, 256, 256)).to(device)
    pseudo_l2 = torch.randint(0, 5, (2, 256, 256)).to(device)

    out = model(O, S)
    loss = criterion(out, pseudo, pseudo_l1, pseudo_l2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[Train-step] loss = {loss.item():.4f}")
    print(f"[Shapes] change={tuple(out['change_prob'].shape)}, "
          f"q1={tuple(out['q1'].shape)}, q2={tuple(out['q2'].shape)}, "
          f"q1_gated={tuple(out['q1_logits_gated'].shape)}, "
          f"transition={tuple(out['transition'].shape)}")
    print("[OK] model.py self-test passed.")


