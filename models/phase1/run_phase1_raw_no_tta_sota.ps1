param(
    [string]$RealData = "C:\AI_PROJECT\PALM_PRINT\data\after\raw_224x224px",
    [string]$PretrainCheckpoint = "",
    [string]$OutputRoot = "runs\phase1_raw_no_tta_sota_v1",
    [int]$Seed = 42,
    [int]$Workers = 8,
    [int]$PkClasses = 8,
    [int]$PkSamples = 4,
    [int]$MinImagesPerSubject = 10,
    [ValidateSet("baseline", "supcon", "supcon_refine")]
    [string]$Variant = "supcon_refine",
    [switch]$RunAll
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$trainScript = Join-Path $PSScriptRoot "train_pvtree.py"
$evalScript = Join-Path $PSScriptRoot "evaluate.py"

if (-not (Test-Path $pythonExe)) {
    throw "Python not found: $pythonExe"
}
if (-not (Test-Path $RealData)) {
    throw "Dataset not found: $RealData"
}

if ([string]::IsNullOrWhiteSpace($PretrainCheckpoint)) {
    $candidateCheckpoints = @(
        (Join-Path $projectRoot "runs\phase1_raw_ultimate_v1\best.pt"),
        (Join-Path $projectRoot "runs\phase1_tongji_v1_raw\best.pt"),
        (Join-Path $projectRoot "runs\phase1_optimal_v3\best.pt")
    )
    foreach ($candidate in $candidateCheckpoints) {
        if (Test-Path $candidate) {
            $PretrainCheckpoint = $candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PretrainCheckpoint) -or -not (Test-Path $PretrainCheckpoint)) {
    throw "No usable pretrain checkpoint found. Pass -PretrainCheckpoint explicitly."
}

function Invoke-NoTtaVariant {
    param(
        [Parameter(Mandatory = $true)][string]$VariantName
    )

    $variantDir = Join-Path $projectRoot $OutputRoot
    $trainOut = Join-Path $variantDir $VariantName
    $evalOut = Join-Path $trainOut "eval_no_tta_primary"

    $epochs = 32
    $freezeEpochs = 6
    $evalEvery = 2
    $patience = 10
    $trainRotateDeg = 25
    $tripletWeight = "0.35"
    $tripletMining = "semi-hard"
    $centerLossWeight = "0.003"
    $arcSEnd = "36.0"
    $arcMEnd = "0.38"
    $supcon = $false
    $supconWeight = "0.10"
    $supconStopEpoch = "10"
    $useMinimalAug = $false
    $useVeinTrAug = $true
    $bestBy = "eer"

    switch ($VariantName) {
        "baseline" {
            $epochs = 28
            $trainRotateDeg = 20
            $tripletWeight = "0.30"
            $centerLossWeight = "0.003"
            $arcSEnd = "34.0"
            $arcMEnd = "0.35"
            $supcon = $false
            $useMinimalAug = $true
            $useVeinTrAug = $false
        }
        "supcon" {
            $epochs = 32
            $trainRotateDeg = 25
            $tripletWeight = "0.35"
            $centerLossWeight = "0.003"
            $arcSEnd = "38.0"
            $arcMEnd = "0.38"
            $supcon = $true
            $supconWeight = "0.10"
            $supconStopEpoch = "10"
            $useMinimalAug = $false
            $useVeinTrAug = $true
        }
        "supcon_refine" {
            $epochs = 40
            $freezeEpochs = 8
            $evalEvery = 2
            $patience = 12
            $trainRotateDeg = 30
            $tripletWeight = "0.40"
            $tripletMining = "hard"
            $centerLossWeight = "0.005"
            $arcSEnd = "48.0"
            $arcMEnd = "0.42"
            $supcon = $true
            $supconWeight = "0.12"
            $supconStopEpoch = "12"
            $useMinimalAug = $false
            $useVeinTrAug = $true
        }
        default {
            throw "Unsupported variant: $VariantName"
        }
    }

    $prefetchFactor = if ($Workers -le 2) { 2 } else { 4 }
    $arcWarmup = [Math]::Min(8, $epochs - 2)
    $tripletWarmup = [Math]::Min(8, $epochs - 2)
    if ($arcWarmup -lt 2) { $arcWarmup = 2 }
    if ($tripletWarmup -lt 2) { $tripletWarmup = 2 }

    $trainCmd = @(
        $trainScript,
        "--real-data", $RealData,
        "--synthetic-root", "unused",
        "--output", $trainOut,
        "--seed", "$Seed",
        "--force-gpu",
        "--amp",
        "--channels-last",
        "--backbone-name", "swin_tiny_patch4_window7_224",
        "--skip-pretrain",
        "--skip-synth-generation",
        "--pretrain-checkpoint", $PretrainCheckpoint,
        "--split-mode", "within_subject",
        "--verification-mode", "train_gallery",
        "--gallery-score-mode", "mean_template",
        "--gallery-probe-znorm",
        "--train-ratio", "0.80",
        "--val-ratio", "0.10",
        "--test-ratio", "0.10",
        "--min-images-per-subject", "$MinImagesPerSubject",
        "--min-eval-images-per-subject", "1",
        "--workers", "$Workers",
        "--prefetch-factor", "$prefetchFactor",
        "--arc-warmup-epochs", "$arcWarmup",
        "--triplet-warmup-epochs", "$tripletWarmup",
        "--arc-s-start", "16.0",
        "--arc-s-end", "$arcSEnd",
        "--arc-m-start", "0.0",
        "--arc-m-end", "$arcMEnd",
        "--label-smoothing", "0.0",
        "--triplet-weight", "$tripletWeight",
        "--triplet-weight-start", "0.05",
        "--triplet-margin", "0.20",
        "--triplet-mining", "$tripletMining",
        "--negative-queue-size", "4096",
        "--center-loss-weight", "$centerLossWeight",
        "--center-loss-lr", "0.5",
        "--pk-classes", "$PkClasses",
        "--pk-samples", "$PkSamples",
        "--steps-per-epoch", "0",
        "--accum-steps", "1",
        "--finetune-epochs", "$epochs",
        "--finetune-freeze-epochs", "$freezeEpochs",
        "--finetune-batch-size", "32",
        "--finetune-lr-freeze", "3e-4",
        "--finetune-lr-backbone", "3e-5",
        "--finetune-lr-head", "3e-5",
        "--finetune-weight-decay", "0.05",
        "--lr-min-factor", "0.01",
        "--emb-dropout", "0.10",
        "--drop-path-rate", "0.20",
        "--ema-decay", "0.9998",
        "--eval-every", "$evalEvery",
        "--patience", "$patience",
        "--best-by", "$bestBy",
        "--save-topk-eer", "10",
        "--use-clahe",
        "--train-rotate-deg", "$trainRotateDeg"
    )

    if ($useMinimalAug) {
        $trainCmd += "--minimal-aug"
    }
    if ($useVeinTrAug) {
        $trainCmd += "--veintr-aug"
    }
    if ($supcon) {
        $trainCmd += @("--supcon", "--supcon-weight", "$supconWeight", "--supcon-temp", "0.07", "--supcon-stop-epoch", "$supconStopEpoch")
    }

    Write-Host ""
    Write-Host "=============================================================="
    Write-Host "  No-TTA SOTA Pipeline"
    Write-Host "=============================================================="
    Write-Host "Variant           : $VariantName"
    Write-Host "Dataset           : $RealData"
    Write-Host "Warm start        : $PretrainCheckpoint"
    Write-Host "Train output      : $trainOut"
    Write-Host "Eval output       : $evalOut"
    Write-Host "Split             : within_subject 8/1/1-style"
    Write-Host "No-TTA target     : yes"
    Write-Host "SupCon            : $supcon"
    Write-Host "Train rotate      : +/-$trainRotateDeg deg"
    Write-Host "=============================================================="

    & $pythonExe @trainCmd
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $bestCheckpoint = Join-Path $trainOut "best.pt"
    if (-not (Test-Path $bestCheckpoint)) {
        throw "Training finished but best checkpoint not found: $bestCheckpoint"
    }

    $evalCmd = @(
        $evalScript,
        "--checkpoint", $bestCheckpoint,
        "--data", $RealData,
        "--output-dir", $evalOut,
        "--batch-size", "32",
        "--split-mode", "within_subject",
        "--verification-mode", "train_gallery",
        "--gallery-score-mode", "mean_template",
        "--gallery-probe-znorm",
        "--train-ratio", "0.80",
        "--val-ratio", "0.10",
        "--test-ratio", "0.10",
        "--min-images-per-subject", "$MinImagesPerSubject",
        "--min-eval-images-per-subject", "1",
        "--no-tta",
        "--amp",
        "--channels-last",
        "--measure-latency",
        "--latency-warmup", "10",
        "--latency-iters", "50",
        "--eval-rotate-deg", "20,30,45,90",
        "--no-plots",
        "--force-gpu"
    )

    & $pythonExe @evalCmd
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host ""
    Write-Host "Completed variant: $VariantName"
    Write-Host "  - checkpoint: $bestCheckpoint"
    Write-Host "  - metrics:    $(Join-Path $evalOut 'evaluation_metrics.json')"
    Write-Host "  - latency:    $(Join-Path $evalOut 'latency_metrics.json')"
    Write-Host "  - rotations:  $(Join-Path $evalOut 'rotation_sweep.json')"
}

$variants = if ($RunAll) {
    @("baseline", "supcon", "supcon_refine")
} else {
    @($Variant)
}

foreach ($name in $variants) {
    Invoke-NoTtaVariant -VariantName $name
}
