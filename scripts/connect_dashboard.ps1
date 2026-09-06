$ErrorActionPreference = 'Stop'
# Keep the V12 dashboard on 8789 and the local V3 dashboard on 8790.
while ($true) {
    ssh -N -i "$env:USERPROFILE/.ssh/id_ed25519_runpod_cftn" -p 12561 -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 0.0.0.0:8791:127.0.0.1:8790 root@103.196.86.190
    Start-Sleep -Seconds 10
}
