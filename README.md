# PalmVeinFormer no-TTA

Repo nay tach phan code lien quan den hai huong PalmVeinFormer tu workspace goc `PALM_PRINT`, va branch hien tai uu tien cho pipeline `no-TTA`.

Pham vi duoc dua vao:
- trainer dung chung: `models/phase1/train_pvtree.py`
- trainer/eval helpers: `models/phase1/train.py`, `models/phase1/evaluate.py`
- model, metrics, transforms, data split, TTA utils
- script chay RAW TTA: `models/phase1/run_phase1_raw_ultimate.ps1`
- script chay RAW no-TTA: `models/phase1/run_phase1_raw_no_tta_sota.ps1`

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
models/phase1/run_phase1_raw_no_tta_sota.ps1
models/phase1/tta_utils.py
```

## Cach chay no-TTA RAW

```powershell
& .\models\phase1\run_phase1_raw_no_tta_sota.ps1 `
  -RealData C:\path\to\raw_224x224px `
  -PretrainCheckpoint C:\path\to\best.pt `
  -Variant supcon_refine
```

Co the chay day du 3 bien the:

```powershell
& .\models\phase1\run_phase1_raw_no_tta_sota.ps1 `
  -RealData C:\path\to\raw_224x224px `
  -PretrainCheckpoint C:\path\to\best.pt `
  -RunAll
```

## Danh gia checkpoint no-TTA

```powershell
python .\models\phase1\evaluate.py `
  --checkpoint C:\path\to\best.pt `
  --data C:\path\to\raw_224x224px `
  --verification-mode train_gallery `
  --gallery-score-mode mean_template `
  --gallery-probe-znorm `
  --no-tta
```

## Neu can branch TTA

Branch TTA van co san o remote:
`tta-only-20260323`
