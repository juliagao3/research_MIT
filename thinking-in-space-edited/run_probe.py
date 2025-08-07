import sys
sys.path.insert(0, "thinking-in-space")

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

walk_dirs = sorted(glob(os.path.join("/content/egocentric_data/full_walk_views", "walk_*")))
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
    train_loss = 0
    test_loss = 0
    probe.train()

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    probe.eval()
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

    print(f"Epoch {epoch}, avg train loss: {train_loss / len(train_loader):.4f}, avg test loss: {test_loss / len(test_loader):.4f}")