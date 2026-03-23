# PalmVeinFormer-TTA

Repo nay chi tach phan code lien quan den nhanh PalmVeinFormer-TTA tu workspace goc `PALM_PRINT`.

Pham vi duoc dua vao:
- trainer TTA: `models/phase1/train_pvtree.py`
- trainer/eval helpers: `models/phase1/train.py`, `models/phase1/evaluate.py`
- model, metrics, transforms, data split, TTA utils
- script chay RAW TTA: `models/phase1/run_phase1_raw_ultimate.ps1`

Khong dua vao repo nay:
- paper/docx
- dataset
- checkpoint `.pt`
- cac nhanh code khac nhu SupCon, GCN, ROI viewer, external tools khong can thiet

## Cai dat

```powershell
pip install -r requirements.txt
```

Luu y: `torch`/`torchvision` CUDA nen cai theo dung CUDA version cua may.

## Cac file chinh

```text
models/phase1/train_pvtree.py
models/phase1/train.py
models/phase1/evaluate.py
models/phase1/run_phase1_raw_ultimate.ps1
models/phase1/tta_utils.py
```

## Cach chay TTA RAW

```powershell
& .\models\phase1\run_phase1_raw_ultimate.ps1 `
  -RealData C:\path\to\raw_224x224px `
  -PretrainCheckpoint C:\path\to\best.pt `
  -Output runs\phase1_raw_ultimate_v1
```

## Danh gia checkpoint

```powershell
python .\models\phase1\evaluate.py `
  --checkpoint C:\path\to\best.pt `
  --data C:\path\to\raw_224x224px `
  --verification-mode train_gallery `
  --gallery-score-mode mean_template `
  --gallery-probe-znorm `
  --tta `
  --tta-variants hflip
```
