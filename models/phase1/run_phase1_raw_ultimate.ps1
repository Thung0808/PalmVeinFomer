param(
    [string]$RealData = "C:\AI_PROJECT\PALM_PRINT\data\after\raw_224x224px",
    [string]$PretrainCheckpoint = "",
    [string]$Output = "runs\phase1_raw_ultimate_v1",
    [int]$Seed = 42,
    [int]$Epochs = 140,
    [int]$FreezeEpochs = 12,
    [int]$Workers = 4,
    [int]$MinImagesPerSubject = 4,
    [ValidateSet("swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224")]
    [string]$BackboneName = "swin_tiny_patch4_window7_224",
    [int]$ArcSubcenters = 1,
    [double]$LabelSmoothing = 0.0,
    [double]$ArcSEnd = 30.0,
    [double]$ArcMEnd = 0.35,
    [double]$TripletWeight = 0.40,
    [ValidateSet("semi-hard", "hard")]
    [string]$TripletMining = "semi-hard",
    [int]$AccumSteps = 2,
    [int]$PkClasses = 8,
    [int]$PkSamples = 4,
    [double]$CenterLossWeight = -1.0,
    [double]$EmaDecay = 0.9998,
    [ValidateSet("eer", "tar1e4", "tar1e5")]
    [string]$BestBy = "eer",
    [switch]$NoClahe,
    [switch]$Refine
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$trainScript = Join-Path $PSScriptRoot "train_pvtree.py"
$tableScript = Join-Path $PSScriptRoot "print_run_table.py"

if (-not (Test-Path $pythonExe)) {
    throw "Python not found: $pythonExe"
}
if (-not (Test-Path $RealData)) {
    throw "Dataset not found: $RealData"
}

if ([string]::IsNullOrWhiteSpace($PretrainCheckpoint)) {
    $candidateCheckpoints = @()
    switch ($BackboneName) {
        "swin_small_patch4_window7_224" {
            $candidateCheckpoints += (Join-Path $projectRoot "runs\phase1_eer02_swin_small_20260305_221639\best.pt")
        }
        default {
            $candidateCheckpoints += (Join-Path $projectRoot "runs\phase1_tongji_v1_raw\best.pt")
            $candidateCheckpoints += (Join-Path $projectRoot "runs\phase1_optimal_v3\best.pt")
        }
    }
    foreach ($candidate in $candidateCheckpoints) {
        if (Test-Path $candidate) {
            $PretrainCheckpoint = $candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PretrainCheckpoint) -or -not (Test-Path $PretrainCheckpoint)) {
    throw "No usable pretrain checkpoint found for backbone $BackboneName. Pass -PretrainCheckpoint explicitly."
}

$subjectDirs = Get-ChildItem $RealData -Directory
$allSubjectCount = $subjectDirs.Count
$allImageCount = ($subjectDirs | ForEach-Object { (Get-ChildItem $_.FullName -File).Count } | Measure-Object -Sum).Sum
$usableSubjectDirs = $subjectDirs | Where-Object { (Get-ChildItem $_.FullName -File).Count -ge $MinImagesPerSubject }
$usableSubjectCount = $usableSubjectDirs.Count
$usableImageCount = ($usableSubjectDirs | ForEach-Object { (Get-ChildItem $_.FullName -File).Count } | Measure-Object -Sum).Sum

if ($usableSubjectCount -le 0) {
    throw "No usable subjects remain after min-images filter ($MinImagesPerSubject)."
}

$avgImagesAll = if ($allSubjectCount -gt 0) { [Math]::Round(($allImageCount / $allSubjectCount), 2) } else { 0.0 }
$avgImagesUsable = if ($usableSubjectCount -gt 0) { [Math]::Round(($usableImageCount / $usableSubjectCount), 2) } else { 0.0 }

$prefetchFactor = if ($Workers -le 2) { 2 } else { 4 }
$arcWarmup = [Math]::Min(10, $Epochs - 2)
$tripletWarmup = [Math]::Min(10, $Epochs - 2)
if ($arcWarmup -lt 2) { $arcWarmup = 2 }
if ($tripletWarmup -lt 2) { $tripletWarmup = 2 }

$arcSEndUserSet = $PSBoundParameters.ContainsKey("ArcSEnd")
$arcMEndUserSet = $PSBoundParameters.ContainsKey("ArcMEnd")
$tripletWeightUserSet = $PSBoundParameters.ContainsKey("TripletWeight")
$tripletMiningUserSet = $PSBoundParameters.ContainsKey("TripletMining")
$accumStepsUserSet = $PSBoundParameters.ContainsKey("AccumSteps")
$bestByUserSet = $PSBoundParameters.ContainsKey("BestBy")

if ($Refine) {
    if (-not $arcSEndUserSet) { $ArcSEnd = 64.0 }
    if (-not $arcMEndUserSet) { $ArcMEnd = 0.50 }
    if (-not $tripletWeightUserSet) { $TripletWeight = 0.45 }
    if (-not $accumStepsUserSet) { $AccumSteps = 4 }
    if (-not $bestByUserSet) { $BestBy = "tar1e5" }
}

$centerLossWeightFinal = if ($CenterLossWeight -ge 0.0) {
    [string]$CenterLossWeight
}
elseif ($Refine) {
    "0.005"
}
else {
    "0.003"
}
$saveTopK = if ($Refine) { "15" } else { "10" }
$patience = if ($Refine) { "40" } else { "35" }
$sortBy = switch ($BestBy) {
    "tar1e4" { "val_tar_1e4" }
    "tar1e5" { "val_tar_1e5" }
    default { "val_eer" }
}

$cmd = @(
    $trainScript,
    "--real-data", $RealData,
    "--synthetic-root", "unused",
    "--output", $Output,
    "--seed", "$Seed",
    "--force-gpu",
    "--amp",
    "--backbone-name", "$BackboneName",
    "--skip-pretrain",
    "--skip-synth-generation",
    "--pretrain-checkpoint", $PretrainCheckpoint,
    "--min-images-per-subject", "$MinImagesPerSubject",
    "--workers", "$Workers",
    "--prefetch-factor", "$prefetchFactor",
    "--train-ratio", "0.80",
    "--val-ratio", "0.10",
    "--test-ratio", "0.10",
    "--arc-warmup-epochs", "$arcWarmup",
    "--triplet-warmup-epochs", "$tripletWarmup",
    "--arc-s-start", "16.0",
    "--arc-s-end", "$ArcSEnd",
    "--arc-m-start", "0.0",
    "--arc-m-end", "$ArcMEnd",
    "--arc-subcenters", "$ArcSubcenters",
    "--label-smoothing", "$LabelSmoothing",
    "--triplet-weight", "$TripletWeight",
    "--triplet-weight-start", "0.05",
    "--triplet-margin", "0.20",
    "--triplet-mining", "$TripletMining",
    "--negative-queue-size", "4096",
    "--center-loss-weight", $centerLossWeightFinal,
    "--center-loss-lr", "0.5",
    "--pk-classes", "$PkClasses",
    "--pk-samples", "$PkSamples",
    "--steps-per-epoch", "0",
    "--accum-steps", "$AccumSteps",
    "--finetune-epochs", "$Epochs",
    "--finetune-freeze-epochs", "$FreezeEpochs",
    "--finetune-batch-size", "32",
    "--finetune-lr-freeze", "3e-4",
    "--finetune-lr-backbone", "3e-5",
    "--finetune-lr-head", "3e-5",
    "--finetune-weight-decay", "0.05",
    "--lr-min-factor", "0.01",
    "--emb-dropout", "0.10",
    "--drop-path-rate", "0.20",
    "--ema-decay", "$EmaDecay",
    "--tta",
    "--cosine-restarts", "1",
    "--best-by", $BestBy,
    "--save-topk-eer", $saveTopK,
    "--patience", $patience
)

if ($Refine) {
    $cmd += "--strong-aug"
}
if (-not $NoClahe) {
    $cmd += "--use-clahe"
}

Write-Host ""
Write-Host "=============================================================="
Write-Host "  Phase1 RAW Ultimate - CLAHE + EMA + TTA + CenterLoss"
Write-Host "=============================================================="
Write-Host "Dataset           : $RealData"
Write-Host "Warm start        : $PretrainCheckpoint"
Write-Host "Output            : $Output"
Write-Host "Best checkpoint by: $BestBy"
Write-Host "Refine mode       : $Refine"
Write-Host ""
Write-Host "Dataset info:"
Write-Host "  - All subjects : $allSubjectCount"
Write-Host "  - All images   : $allImageCount (avg $avgImagesAll/subject)"
Write-Host "  - Usable after filter >= $MinImagesPerSubject images: $usableSubjectCount subjects, $usableImageCount images (avg $avgImagesUsable/subject)"
Write-Host ""
Write-Host "Training profile:"
Write-Host "  - Backbone: $BackboneName"
Write-Host "  - Split: 80/10/10 subject-independent"
Write-Host "  - PK sampler: P=$PkClasses, K=$PkSamples, accum=$AccumSteps"
Write-Host "  - CLAHE: $([bool](-not $NoClahe)) | EMA + TTA enabled"
Write-Host "  - ArcFace target: s=$ArcSEnd, m=$ArcMEnd, subcenters=$ArcSubcenters"
Write-Host "  - Label smoothing: $LabelSmoothing"
Write-Host "  - Triplet: weight=$TripletWeight, mining=$TripletMining"
Write-Host "  - EMA decay: $EmaDecay"
Write-Host "  - CenterLoss weight: $centerLossWeightFinal"
if ($Refine) {
    Write-Host "  - Strong augmentation enabled for harder low-FAR refinement"
    if (-not $tripletMiningUserSet -and $TripletMining -eq "semi-hard") {
        Write-Host "  - Note: trainer will auto-switch semi-hard -> hard after triplet warmup"
    }
}
Write-Host ""

& $pythonExe @cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Training complete. Horizontal log table:"
& $pythonExe $tableScript --run-dir (Resolve-Path $Output) --sort-by $sortBy --topk 10
exit $LASTEXITCODE
