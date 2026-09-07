param([string]$Python='C:/Users/adria/anaconda3/python.exe', [string]$ConfigFile='config/local_curriculum_v3.json')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$configPath=Join-Path $projectRoot $ConfigFile
$config=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$running=Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match '-m cftn_v3\.(local_math_training|full_curriculum_training|local_curriculum_pipeline)'
}
if ($running) { throw 'A local training worker or pipeline is already running. Inspect it before starting another.' }
if ((Test-Path -LiteralPath "$($config.root)/pipeline.lock") -or (Test-Path -LiteralPath "$($config.root)/native_training.lock")) {
    throw 'A pipeline lock exists. Verify its recorded process before clearing a stale lock.'
}
New-Item -ItemType Directory -Force -Path $config.root | Out-Null
$job=Start-Process -FilePath $Python -ArgumentList '-u','-m','cftn_v3.local_curriculum_pipeline','--config',$configPath `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$($config.root)/pipeline.stdout.log" -RedirectStandardError "$($config.root)/pipeline.stderr.log"
$job.Id | Set-Content -LiteralPath "$($config.root)/pipeline.pid"
Write-Output "Local curriculum pipeline PID: $($job.Id). Artifacts: $($config.root)"
