param(
    [Parameter(Mandatory = $true)][ValidateSet('dataset804_probtab_t0_p3rfs_v1', 'dataset804_807_refine_emptytiny_t0_804p3rfs_v1')][string]$Variant,
    [Parameter(Mandatory = $true)][string]$CaseId,
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OriginalTaskDir,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$input = (Resolve-Path $InputDir).Path
$originalTask = (Resolve-Path $OriginalTaskDir).Path
$originalModel = Join-Path $originalTask 'model'
$repoModel = Join-Path $repo ("weights\{0}\model" -f $Variant)
$outputRoot = [IO.Path]::GetFullPath($OutputRoot)
$referenceOutput = Join-Path $outputRoot "$CaseId-original-docker-source"
$repositoryOutput = Join-Path $outputRoot "$CaseId-repository-source"
$runtime = Join-Path $repo 'local_testdata\runtime313'

foreach ($dir in @($outputRoot, $referenceOutput, $repositoryOutput)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}
foreach ($dir in @('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')) {
    New-Item -ItemType Directory -Path (Join-Path $runtime $dir) -Force | Out-Null
}

$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:nnUNet_raw = Join-Path $runtime 'nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $runtime 'nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $runtime 'nnUNet_results'
$env:HECKTOR_ALLOW_BATCH_ENTRY = '0'
$env:HECKTOR_BATCH_MODE = 'sequential'
$env:HECKTOR_CPU_WORKERS = '1'
$env:HECKTOR_LOCALROI_WORKERS = '1'
$env:HECKTOR_POST_WORKERS = '0'
$env:HECKTOR_PREFETCH = '1'
$env:HECKTOR_COARSE_BATCH_SIZE = '1'
$env:HECKTOR_FINE_TILE_STEP_SIZE = '0.8'
$env:HECKTOR_FINE_TILE_BATCH_SIZE = '1'
$env:HECKTOR_FINE_USE_TTA = '1'
$env:HECKTOR_FINE_FOLDS = '0,1,2,3,4'

$runner = Join-Path $repo 'tools\run_local.py'
$comparator = Join-Path $repo 'tools\compare_outputs.py'
Write-Host "Running original Docker Task source: $CaseId"
& $PythonExe $runner $Variant $input $referenceOutput --source-root $originalTask --model-root $originalModel
if ($LASTEXITCODE -ne 0) { throw "Original Docker Task source failed for $CaseId (exit $LASTEXITCODE)" }

Write-Host "Running repository source: $CaseId"
& $PythonExe $runner $Variant $input $repositoryOutput --model-root $repoModel
if ($LASTEXITCODE -ne 0) { throw "Repository source failed for $CaseId (exit $LASTEXITCODE)" }

Write-Host "Comparing segmentation and task outputs: $CaseId"
& $PythonExe $comparator $referenceOutput $repositoryOutput
if ($LASTEXITCODE -ne 0) { throw "Output mismatch for $CaseId" }

Write-Host "Parity passed for $Variant / $CaseId"
