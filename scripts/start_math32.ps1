param([string]$Python='C:/Users/adria/anaconda3/python.exe')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$running=Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '-m (math32.train|cftn_v3.(criterion_curriculum_training|local_curriculum_pipeline|generated_correction_probe))'}
if($running){throw 'A training worker is already running'}
$artifactRoot='G:/ctfn-text/artifacts/v3_2'
if(Test-Path "$artifactRoot/native_training.lock"){throw 'Inspect existing V3.2 lock before restarting'}
New-Item -ItemType Directory -Force $artifactRoot | Out-Null
$job=Start-Process $Python -ArgumentList '-u','-m','math32.train','--config','config/math32.json' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$artifactRoot/training.stdout.log" -RedirectStandardError "$artifactRoot/training.stderr.log"
$job.Id | Set-Content "$artifactRoot/training.pid"
Write-Output "V3.2 worker PID: $($job.Id)"
