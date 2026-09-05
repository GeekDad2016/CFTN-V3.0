param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [Parameter(Mandatory=$true)][string]$GitRemote,
    [Parameter(Mandatory=$true)][string]$Revision,
    [int]$Port = 22,
    [string]$User = 'root',
    [string]$KeyPath = 'C:\Users\adria\.ssh\id_ed25519_runpod_cftn'
)
$ErrorActionPreference = 'Stop'
if ($SshHost -notmatch '^[a-zA-Z0-9.-]+$' -or $User -notmatch '^[a-zA-Z0-9_-]+$' -or $Revision -notmatch '^[a-fA-F0-9]{40}$') { throw 'Invalid host, user, or exact Git commit' }
if ($GitRemote -notmatch '^(https://|git@)[a-zA-Z0-9./:@_-]+$') { throw 'Unsupported Git remote format' }
$remoteCommand = "set -eu; mkdir -p /workspace; if [ ! -d /workspace/V3.0/.git ]; then git clone '$GitRemote' /workspace/V3.0; fi; cd /workspace/V3.0; test -z `"`$(git status --porcelain)`"; git fetch origin; git checkout --detach '$Revision'; bash scripts/runpod_setup.sh"
& ssh -i $KeyPath -p $Port "$User@$SshHost" $remoteCommand
if ($LASTEXITCODE -ne 0) { throw 'RunPod deployment failed' }
