import sys
sys.path.insert(0, "thinking-in-space-edited")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torch.optim as optim
import pandas as pd
from glob import glob
from PIL import Image
from lmms_eval.models.llava_onevision import Llava_OneVision
import os
import re
from scipy.stats import kendalltau

model = Llava_OneVision(
    pretrained="lmms-lab/llava-onevision-qwen2-0.5b-ov",
    model_name="llava_qwen",
    conv_template="qwen_1_5",
    device="cuda:0"
)

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)

#load the dataset
class WalksDataset(Dataset):
    def __init__(self, walk_dir, transform=None):
        self.samples = sorted(walk_dir)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        walk = self.samples[index]
        dist = pd.read_csv(os.path.join(walk, "distance_matrix.csv"), index_col=0)
        dist = torch.tensor(dist.values, dtype=torch.float32)

        img_files = glob(os.path.join(walk, "walk_step_*.png"))
        img_files = sorted(img_files, key=self.extract_step_num)

        images = [Image.open(fn).convert("RGB") for fn in img_files]
        imgs = torch.stack([self.transform(img) for img in images])

        return imgs, dist

    @staticmethod
    def extract_step_num(filename):
        match = re.search(r'walk_step_(\d+)_', os.path.basename(filename))
        return int(match.group(1)) if match else -1


def upper_tri_vector(M):
    i, j = torch.triu_indices(M.size(0), M.size(1), offset=1, device=M.device)
    return M[i, j]

def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value

def hsic_biased(K, L):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
    """Computes the unbiased Centered Kernel Alignment (CKA) between features."""
    
    # if kernel_metric == 'ip':
    #     # Compute kernel matrices for the linear case
    #     K = torch.mm(feats_A, feats_A.T)
    #     L = torch.mm(feats_B, feats_B.T)
    # elif kernel_metric == 'rbf':
    #     # COMPUTES RBF KERNEL
    #     K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
    #     L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
    # else:
    #     raise ValueError(f"Invalid kernel metric {kernel_metric}")

    # Compute HSIC values
    K = feats_A
    L = feats_B

    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    # Ensure that the minimum value is 0
    hsic_kk = torch.clip(hsic_fn(K, K), 0)
    hsic_ll = torch.clip(hsic_fn(L, L), 0)
    hsic_kl = torch.clip(hsic_fn(K, L), 0)

    # Compute CKA
    #print('hsic', hsic_kl)
    denom = torch.sqrt(hsic_kk * hsic_ll)
    if denom == 0:
        return 0
    cka_value = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)   
    return cka_value.item()

def correlation_score(pdist, gdist):
    pred_flat = upper_tri_vector(pdist)
    gt_flat = upper_tri_vector(gdist)

    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()

    corr = torch.sum(pred_centered * gt_centered) / (torch.sqrt(
      torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(gt_centered ** 2)))

    return corr.item()

def kendalltau_score(pdist, gdist):
    pred = upper_tri_vector(pdist).detach().float().cpu().numpy()
    gt = upper_tri_vector(gdist).detach().float().cpu().numpy()

    tau, _ = kendalltau(pred, gt)   
    return tau

walk_dirs = sorted(glob(os.path.join("/content/research_MIT/egocentric_data/full_walk_views", "walk_*")))
test_dir = walk_dirs[-1:]
train_dirs = walk_dirs[:-1]

train_data = WalksDataset(train_dirs)
test_data = WalksDataset(test_dir)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
probe = LinearProbe(input_dim=896, output_dim=896).to(device)

optimizer = optim.Adam(probe.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

to_pil = transforms.ToPILImage()
for epoch in range(20):
    probe.train()
    train_loss = 0.0
    train_cka_total = 0.0
    train_corr_total = 0.0
    train_tau_total = 0.0

    for imgs, dist in train_loader:
        imgs = imgs[0]
        dist = dist[0].to(device)
        dist = dist / dist.max()

        pil_imgs = [to_pil(img) for img in imgs]

        with torch.inference_mode():
            emb = model.extract_hidden_states(pil_imgs)

        proj = probe(emb.float())
        proj = nn.functional.normalize(proj, p=2, dim=1)

        pdist = torch.cdist(proj, proj, p=2)
        loss = criterion(pdist, dist)

        train_cka_total += cka(pdist, dist)
        train_corr_total += correlation_score(pdist, dist)
        train_tau_total += kendalltau_score(pdist, dist)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    probe.eval()
    test_loss = 0.0
    test_cka_total = 0.0
    test_corr_total = 0.0
    test_tau_total = 0.0

    with torch.no_grad():
        for test_imgs, test_dist in test_loader:
            test_imgs = test_imgs[0]
            test_dist = test_dist[0].to(device)
            test_dist = test_dist / test_dist.max()

            test_pil_imgs = [to_pil(img) for img in test_imgs]

            test_emb = model.extract_hidden_states(test_pil_imgs)

            test_proj = probe(test_emb.float())
            test_proj = nn.functional.normalize(test_proj, p=2, dim=1)

            test_pdist = torch.cdist(test_proj, test_proj, p=2)
            test_loss += criterion(test_pdist, test_dist).item()

            test_cka_total  += cka(test_pdist, test_dist)
            test_corr_total += correlation_score(test_pdist, test_dist)
            test_tau_total  += kendalltau_score(test_pdist, test_dist)

    num_train_batches = len(train_loader)
    num_test_batches = len(test_loader)

    print(f"Epoch {epoch}, "
          f"Train Loss: {train_loss / num_train_batches:.4f}, "
          f"Test Loss: {test_loss / num_test_batches:.4f}, "
          f"Train CKA: {train_cka_total / num_train_batches:.4f}, "
          f"Test CKA: {test_cka_total / num_test_batches:.4f}, "
          f"Train Corr Score: {train_corr_total / num_train_batches:.4f}, "
          f"Test Corr Score: {test_corr_total / num_test_batches:.4f}, "
          f"Train Kendallτ: {train_tau_total / num_train_batches:.4f}, "
          f"Test Kendallτ: {test_tau_total / num_test_batches:.4f}")