param(
    [string]$Python='C:/Users/adria/anaconda3/python.exe',
    [string]$Output='G:/ctfn-text/artifacts/v3_local_math_v1',
    [string]$Dataset='G:/ctfn-text/data/v3_local_math_v1'
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $Output | Out-Null
if(Test-Path -LiteralPath "$Output/training.lock"){throw 'Training lock exists. Inspect the recorded process before restarting.'}
$arguments=@('-u','-m','cftn_v3.local_math_training','--initial-checkpoint',
 'C:/CFTN/artifacts/math_master_experiment_v12/run/checkpoint_epoch_0150.pth',
 '--config','config/local_math_v1.json','--data','C:/CFTN/.datasets/math_master_experiment_v12',
 '--dataset-output',$Dataset,'--output',$Output)
$job=Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
 -RedirectStandardOutput "$Output/training.stdout.log" -RedirectStandardError "$Output/training.stderr.log"
$job.Id | Set-Content "$Output/training.pid"
Write-Output "Local Maths worker: $($job.Id). Output: $Output"
