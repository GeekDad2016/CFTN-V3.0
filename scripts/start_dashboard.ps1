$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$artifactRoot = Join-Path $projectRoot 'artifacts'
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$pythonPath = 'C:\Users\adria\anaconda3\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$dashboardProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u','-m','cftn_v3.cli','serve','--root','artifacts','--host','127.0.0.1','--port','8790') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $artifactRoot 'dashboard.stdout.log') -RedirectStandardError (Join-Path $artifactRoot 'dashboard.stderr.log')
$dashboardProcess.Id | Set-Content (Join-Path $artifactRoot 'dashboard.pid')
Write-Output "V3.0 dashboard PID $($dashboardProcess.Id): http://127.0.0.1:8790"
