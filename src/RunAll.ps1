# ============================================================
# IJF Reproducibility Package
# Run All Scripts
# ============================================================

function Run-Step {
    param (
        [string]$StepName,
        [string]$ScriptPath
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $StepName -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $StartTime = Get-Date

    python $ScriptPath

    $ExitCode = $LASTEXITCODE

    $EndTime = Get-Date
    $ElapsedTime = $EndTime - $StartTime

    if ($ExitCode -ne 0)
    {
        Write-Host ""
        Write-Host "$StepName FAILED." -ForegroundColor Red
        Write-Host ("Elapsed time: {0:hh\:mm\:ss}" -f $ElapsedTime) -ForegroundColor Red
        Write-Host "Execution terminated." -ForegroundColor Red
        exit $ExitCode
    }

    Write-Host ""
    Write-Host "$StepName COMPLETED SUCCESSFULLY." -ForegroundColor Green
    Write-Host ("Elapsed time: {0:hh\:mm\:ss}" -f $ElapsedTime) -ForegroundColor Green
}

$TotalStartTime = Get-Date
Run-Step "Step 1 : Configuration Check" "./Scripts/RELEASE.ConfigCheck.py"
Run-Step "Step 2 : Run Data Analyasis" "./Scripts/RELEASE.DataQuality.py"
Run-Step "Step 3 : Run All Models" "./Scripts/RELEASE.RunModels.py"
Run-Step "Step 4 : Run Sensitivity Analysis" "./Scripts/RELEASE.RunSensitivity.py"
Run-Step "Step 5 : Run Figure and Table" "./Scripts/RELEASE.FigureTable.py"

$TotalEndTime = Get-Date
$TotalElapsedTime = $TotalEndTime - $TotalStartTime

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "ALL TASKS COMPLETED SUCCESSFULLY." -ForegroundColor Green
Write-Host ("Total elapsed time: {0:hh\:mm\:ss}" -f $TotalElapsedTime) -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""