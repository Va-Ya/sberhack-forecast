# Run manually in Administrator PowerShell after reviewing reports/environment.md.
# Refuses to interrupt an existing WSL VM.
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell as Administrator, then run this script.'
}
$activeVMs = @(Get-Process -Name vmmemWSL,vmmem -ErrorAction SilentlyContinue)
if ($activeVMs.Count -gt 0) {
    throw 'A WSL VM is active. Close its workloads yourself before restarting WSL.'
}
Restart-Service -Name WslService -ErrorAction Stop
docker desktop start --detach
Write-Output 'WSL restarted. Check docker version after Docker Desktop finishes starting.'
