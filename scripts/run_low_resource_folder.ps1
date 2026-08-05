param(
    [Parameter(Mandatory = $true)]
    [string]$Images,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$UpstreamOcrRoot = "D:\workplace\opencode\try_comic",

    [ValidateRange(0, 256)]
    [int]$CpuCores = 2,

    [ValidateSet("cpu", "cuda")]
    [string]$OcrDevice = "cpu",

    [switch]$RestartSakuraAfterOcr,

    [string]$SakuraPython = "D:\workplace\opencode\try_comic\venv\Scripts\python.exe",

    [string]$SakuraScript = "D:\workplace\opencode\try_comic\sakura\sakura_server.py",

    [string]$SakuraDllDirectory = "D:\workplace\opencode\try_comic\sakura\llama"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$imagesPath = [System.IO.Path]::GetFullPath($Images)
$outputPath = [System.IO.Path]::GetFullPath($Output)
$ocrRoot = [System.IO.Path]::GetFullPath($UpstreamOcrRoot)
$ocrScript = Join-Path $ocrRoot "background_ocr.py"
$ocrPython = Join-Path $ocrRoot "venv\Scripts\python.exe"
$pipelinePython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$batchScript = Join-Path $projectRoot "batch.py"
$sakuraLauncher = Join-Path $projectRoot "scripts\sakura_compat_launcher.py"
$workPath = Join-Path $outputPath "_work"
$ocrOutput = Join-Path $workPath "ocr"
$logPath = Join-Path $outputPath "_logs"
$statusPath = Join-Path $outputPath "job.status.json"
$startedAt = [DateTimeOffset]::Now.ToString("o")

if (-not (Test-Path -LiteralPath $imagesPath -PathType Container)) {
    throw "Image directory does not exist: $imagesPath"
}
foreach ($required in ($ocrScript, $ocrPython, $pipelinePython, $batchScript, $sakuraLauncher)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required executable or script does not exist: $required"
    }
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
New-Item -ItemType Directory -Path $workPath -Force | Out-Null
New-Item -ItemType Directory -Path $ocrOutput -Force | Out-Null
New-Item -ItemType Directory -Path $logPath -Force | Out-Null

$currentProcess = Get-Process -Id $PID
$availableCores = [Environment]::ProcessorCount
$normalResources = $CpuCores -eq 0
$currentProcess.PriorityClass = if ($normalResources) {
    [System.Diagnostics.ProcessPriorityClass]::Normal
} else {
    [System.Diagnostics.ProcessPriorityClass]::BelowNormal
}
$limitedCores = if ($normalResources) {
    $availableCores
} else {
    [Math]::Max(1, [Math]::Min($CpuCores, $availableCores))
}
$affinityMask = ([int64]1 -shl $limitedCores) - 1
$currentProcess.ProcessorAffinity = [IntPtr]$affinityMask

$env:OMP_NUM_THREADS = "$limitedCores"
$env:OPENBLAS_NUM_THREADS = "$limitedCores"
$env:MKL_NUM_THREADS = "$limitedCores"
$env:NUMEXPR_NUM_THREADS = "$limitedCores"
$env:OPENCV_FOR_THREADS_NUM = "$limitedCores"
$env:PADDLE_NUM_THREADS = "$limitedCores"
$env:CUDA_VISIBLE_DEVICES = if ($OcrDevice -eq "cuda") { "0" } else { "-1" }
$env:TOKENIZERS_PARALLELISM = "false"

$imageExtensions = @(".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
$imageFiles = @(
    Get-ChildItem -LiteralPath $imagesPath -File |
        Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object Name
)
if ($imageFiles.Count -eq 0) {
    throw "No supported images found in: $imagesPath"
}

function Write-JobStatus {
    param(
        [string]$Phase,
        [string]$State,
        [string]$Message,
        [int]$PendingOcr = 0,
        [int]$Completed = 0,
        [int]$Failed = 0
    )
    $payload = [ordered]@{
        artifact = "manga-localizer-background-job"
        artifact_version = 1
        started_at = $startedAt
        updated_at = [DateTimeOffset]::Now.ToString("o")
        state = $State
        phase = $Phase
        message = $Message
        images = $imageFiles.Count
        pending_ocr = $PendingOcr
        completed = $Completed
        failed = $Failed
        images_directory = $imagesPath
        output_directory = $outputPath
        ocr_directory = $ocrOutput
        resource_limits = [ordered]@{
            priority = $currentProcess.PriorityClass.ToString()
            logical_cores = $limitedCores
            affinity_mask = "0x{0:X}" -f $affinityMask
            library_threads = $limitedCores
            ocr_device = $OcrDevice
            paddle_device = "cpu"
            sequential_phases = $true
        }
        manual_review = [ordered]@{
            mode = "bypassed"
            risks_preserved = $true
        }
        layout = [ordered]@{
            soft_limit_percent = 10
            hard_limit_percent = 20
            preserve_unit_center = $true
        }
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Test-SakuraHealthy {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -TimeoutSec 3
        return $health.status -eq "ok"
    }
    catch {
        return $false
    }
}

function Ensure-SakuraRunning {
    if (-not $RestartSakuraAfterOcr -or (Test-SakuraHealthy)) {
        return
    }
    foreach ($required in ($SakuraPython, $SakuraScript, $SakuraDllDirectory)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            if ($required -eq $SakuraDllDirectory -and (Test-Path -LiteralPath $required -PathType Container)) {
                continue
            }
            throw "Cannot restart Sakura; missing: $required"
        }
    }
    $sakuraStdout = Join-Path $logPath "sakura.stdout.log"
    $sakuraStderr = Join-Path $logPath "sakura.stderr.log"
    Remove-Item Env:LLAMA_CPP_LIB_PATH -ErrorAction SilentlyContinue
    Start-Process `
        -FilePath $SakuraPython `
        -ArgumentList "`"$sakuraLauncher`" `"$SakuraScript`" `"$SakuraDllDirectory`"" `
        -WorkingDirectory (Split-Path -Parent $SakuraScript) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $sakuraStdout `
        -RedirectStandardError $sakuraStderr | Out-Null
    $deadline = [DateTimeOffset]::Now.AddMinutes(3)
    while ([DateTimeOffset]::Now -lt $deadline) {
        Start-Sleep -Seconds 2
        if (Test-SakuraHealthy) {
            return
        }
    }
    throw "Sakura did not become healthy within three minutes"
}

try {
    $missing = @(
        $imageFiles | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $ocrOutput "$($_.BaseName).ocr.json") -PathType Leaf)
        }
    )
    if ($missing.Count -gt 0) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $ocrInput = Join-Path $workPath "ocr-input-$stamp"
        New-Item -ItemType Directory -Path $ocrInput -Force | Out-Null
        foreach ($image in $missing) {
            Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $ocrInput $image.Name)
        }
        Write-JobStatus -Phase "ocr" -State "running" -Message "Generating schema 4 OCR" -PendingOcr $missing.Count
        $ocrLog = Join-Path $logPath "ocr.log"
        $savedErrorPreference = $ErrorActionPreference
        try {
            # PowerShell 5 wraps any native stderr line as an ErrorRecord. MangaOCR
            # writes normal model-loading progress there, so only the exit code is authoritative.
            $ErrorActionPreference = "Continue"
            & $ocrPython $ocrScript $ocrInput $ocrOutput --device $OcrDevice --paddle-device cpu --no-manual-review 2>&1 |
                Tee-Object -FilePath $ocrLog -Append
            $ocrExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedErrorPreference
        }
        if ($ocrExit -ne 0) {
            throw "OCR process exited with code $ocrExit"
        }
    }

    $remaining = @(
        $imageFiles | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $ocrOutput "$($_.BaseName).ocr.json") -PathType Leaf)
        }
    )
    if ($remaining.Count -gt 0) {
        throw "OCR did not produce JSON for $($remaining.Count) image(s)"
    }

    if ($RestartSakuraAfterOcr) {
        Write-JobStatus -Phase "translator-startup" -State "running" -Message "Restarting Sakura after GPU OCR"
        Ensure-SakuraRunning
    }
    Write-JobStatus -Phase "translation" -State "running" -Message "Running latest Manga Localizer pipeline"
    $batchLog = Join-Path $logPath "batch.log"
    Push-Location $projectRoot
    try {
        $savedErrorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $pipelinePython $batchScript `
                --images $imagesPath `
                --ocr-dir $ocrOutput `
                --output-dir $outputPath `
                --translator sakura `
                --translation-base-url "http://127.0.0.1:8080/v1" `
                --translation-model "sukinishiro" `
                --translation-timeout 120 `
                --translation-attempts 2 `
                --translation-prompt-style auto `
                --no-manual-review `
                --max-layout-overflow 20 `
                --preserve-unit-center 2>&1 |
                Tee-Object -FilePath $batchLog -Append
            $batchExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedErrorPreference
        }
    }
    finally {
        Pop-Location
    }

    $reportPath = Join-Path $outputPath "batch.report.json"
    $completed = 0
    $failed = $imageFiles.Count
    if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
        $report = Get-Content -Raw -LiteralPath $reportPath -Encoding utf8 | ConvertFrom-Json
        $completed = [int]$report.summary.completed
        $failed = [int]$report.summary.failed
    }
    $finalState = if ($batchExit -eq 0) { "completed" } else { "completed_with_failures" }
    Write-JobStatus -Phase "finished" -State $finalState -Message "Background batch finished" -Completed $completed -Failed $failed
    exit $batchExit
}
catch {
    $failureMessage = $_.Exception.Message
    if ($RestartSakuraAfterOcr) {
        try {
            Ensure-SakuraRunning
        }
        catch {
            $failureMessage = "$failureMessage; Sakura restart also failed: $($_.Exception.Message)"
        }
    }
    Write-JobStatus -Phase "failed" -State "failed" -Message $failureMessage
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $logPath "job.error.log") -Encoding utf8
    throw
}
