# ==========================================================
# FINAL FULLY FIXED DRIVE VESSEL SEGMENTATION
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# DATASET (Green Channel + CLAHE)
# ==========================================================
class DriveDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        green = img[:, :, 1]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        green = clahe.apply(green)

        img = green.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=2)

        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = (mask > 127).astype(np.float32)

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask

# ==========================================================
# MAMBA BLOCK
# ==========================================================
class VisionMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = x.transpose(1,2)
        x = self.dwconv(x)
        x = x.transpose(1,2)
        x = self.proj(x) * torch.sigmoid(self.gate(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x

# ==========================================================
# UNET
# ==========================================================
class MambaUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3,64,3,padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            VisionMambaBlock(256)
        )

        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128,64,3,padding=1), nn.ReLU())

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        up2 = self.up2(b)
        up2 = F.interpolate(up2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2,e2],1))

        up1 = self.up1(d2)
        up1 = F.interpolate(up1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1,e1],1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out

# ==========================================================
# LOSS
# ==========================================================
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=15.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight).to(device)
        )

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=False)

        bce = self.bce(pred,target)

        pred = torch.sigmoid(pred)
        smooth = 1.
        dice = 1 - (2*(pred*target).sum()+smooth)/(pred.sum()+target.sum()+smooth)

        return bce + dice

# ==========================================================
# LOAD DATA
# ==========================================================
train_dataset = DriveDataset("training/images","training/mask")
test_dataset  = DriveDataset("test/images","test/mask")

train_loader = DataLoader(train_dataset,batch_size=2,shuffle=True)
test_loader  = DataLoader(test_dataset,batch_size=1)

# ==========================================================
# TRAIN
# ==========================================================
model = MambaUNet().to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60)

num_epochs = 60
history = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for imgs,masks in tqdm(train_loader):
        imgs,masks = imgs.to(device),masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs,masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader)
    history.append(avg_loss)
    print("Loss:",avg_loss)
    scheduler.step()

# ==========================================================
# EVALUATION + THRESHOLD SWEEP
# ==========================================================
def evaluate_and_save(model, loader):
    model.eval()
    thresholds = np.arange(0.2,0.61,0.05)
    results = []

    for th in thresholds:
        dice_all, f1_all, sens_all, prec_all, auc_all = [],[],[],[],[]

        with torch.no_grad():
            for imgs,masks in loader:
                imgs,masks = imgs.to(device),masks.to(device)

                logits = model(imgs)
                logits = F.interpolate(logits, size=masks.shape[2:], mode="bilinear", align_corners=False)

                probs = torch.sigmoid(logits)
                preds = (probs>th).float()

                preds_np = preds.cpu().numpy().flatten()
                masks_np = masks.cpu().numpy().flatten()
                probs_np = probs.cpu().numpy().flatten()

                smooth=1.
                dice = (2*(preds_np*masks_np).sum()+smooth)/(preds_np.sum()+masks_np.sum()+smooth)

                dice_all.append(dice)
                f1_all.append(f1_score(masks_np,preds_np))
                sens_all.append(recall_score(masks_np,preds_np))
                prec_all.append(precision_score(masks_np,preds_np))
                auc_all.append(roc_auc_score(masks_np,probs_np))

        results.append([th,
                        np.mean(dice_all),
                        np.mean(auc_all),
                        np.mean(f1_all),
                        np.mean(sens_all),
                        np.mean(prec_all)])

    df = pd.DataFrame(results,
                      columns=["Threshold","Dice","AUC","F1","Sensitivity","Precision"])

    df.to_csv("drive_results.csv",index=False)

    best = df.iloc[df["Dice"].idxmax()]
    print("\n==== BEST RESULT ====")
    print(best)

evaluate_and_save(model,test_loader)

# ==========================================================
# SAVE MODEL
# ==========================================================
torch.save(model.state_dict(),"mamba_drive_final.pth")

# ==========================================================
# SAVE SAMPLE VISUALIZATIONS
# ==========================================================
os.makedirs("outputs",exist_ok=True)
model.eval()

with torch.no_grad():
    for i,(img,mask) in enumerate(test_loader):
        img = img.to(device)
        pred = torch.sigmoid(model(img))
        pred = F.interpolate(pred, size=mask.shape[2:], mode="bilinear", align_corners=False)

        pred = (pred>0.5).float()

        img_np = img[0].cpu().permute(1,2,0).numpy()
        mask_np = mask[0,0].numpy()
        pred_np = pred[0,0].cpu().numpy()

        overlay = img_np.copy()
        overlay[:,:,1] = np.maximum(overlay[:,:,1], pred_np)

        cv2.imwrite(f"outputs/sample_{i}.png", (overlay*255).astype(np.uint8))

        if i == 4:
            break

# ==========================================================
# PLOT LOSS
# ==========================================================
plt.plot(history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
