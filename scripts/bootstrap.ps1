param(
    [switch]$InstallWsl,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $Root ".verifier-runtime"
$Cache = Join-Path $Runtime "cache"
$Log = Join-Path $Runtime "bootstrap.log"
$LockPath = Join-Path $Root "runtime\bootstrap.lock.json"
New-Item -ItemType Directory -Force -Path $Runtime,$Cache | Out-Null
$Lock = Get-Content -Raw -LiteralPath $LockPath | ConvertFrom-Json

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f ([DateTime]::UtcNow.ToString("o")), $Message
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
    Write-Host $Message
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Native([string]$File, [string[]]$Arguments, [switch]$AllowFailure) {
    Write-Log ("exec: {0} {1}" -f $File, ($Arguments -join " "))
    & $File @Arguments
    $code = $LASTEXITCODE
    if (-not $AllowFailure -and $code -ne 0) {
        throw "$File exited with code $code"
    }
    return $code
}

function Test-WslReady {
    try {
        $p = Start-Process -FilePath "wsl.exe" -ArgumentList "--version" -Wait -PassThru -WindowStyle Hidden
        return $p.ExitCode -eq 0
    } catch {
        return $false
    }
}

function Register-Resume {
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Resume"
    $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce"
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -Path $key -Name "NCPAVerifierBootstrap" -Value $cmd -PropertyType String -Force | Out-Null
}

if ($InstallWsl) {
    if (-not (Test-Admin)) { throw "WSL installation requires administrator approval" }
    Write-Log "Enabling/updating WSL 2"
    Invoke-Native "wsl.exe" @("--install", "--no-distribution") -AllowFailure | Out-Null
    Invoke-Native "wsl.exe" @("--update") -AllowFailure | Out-Null
    if (Test-WslReady) { exit 0 }
    exit 3010
}

function Ensure-Wsl {
    if (Test-WslReady) {
        Invoke-Native "wsl.exe" @("--update") -AllowFailure | Out-Null
        return
    }
    Write-Log "WSL 2 is not ready; requesting one-time administrator approval"
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-InstallWsl")
    $p = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList -Wait -PassThru
    if ($p.ExitCode -eq 3010 -or -not (Test-WslReady)) {
        Register-Resume
        Write-Log "Windows restart is required. Setup will resume automatically after sign-in."
        shutdown.exe /r /t 30 /c "NCPA Verifier setup requires one Windows restart. Setup will resume automatically after sign-in."
        exit 0
    }
}

function Ensure-PortablePython {
    $PythonDir = Join-Path $Runtime "python"
    $PythonExe = Join-Path $PythonDir "python.exe"
    if (Test-Path -LiteralPath $PythonExe) { return $PythonExe }

    $Zip = Join-Path $Cache ("python-{0}-embed-amd64.zip" -f $Lock.python.version)
    Write-Log ("Downloading pinned Python {0}" -f $Lock.python.version)
    Invoke-WebRequest -Uri $Lock.python.url -OutFile $Zip -UseBasicParsing
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Zip).Hash.ToLowerInvariant()
    $expected = ([string]$Lock.python.sha256).ToLowerInvariant()
    if ($actual -ne $expected) { throw "Python archive SHA-256 mismatch" }
    New-Item -ItemType Directory -Force -Path $PythonDir | Out-Null
    Expand-Archive -LiteralPath $Zip -DestinationPath $PythonDir -Force
    $pth = Get-ChildItem -LiteralPath $PythonDir -Filter "python*._pth" | Select-Object -First 1
    if (-not $pth) { throw "embedded Python path file not found" }
    $lines = Get-Content -LiteralPath $pth.FullName
    if ($lines -notcontains "..\..\src") {
        Add-Content -LiteralPath $pth.FullName -Value "..\..\src" -Encoding ASCII
    }
    return $PythonExe
}

function Find-DockerExe {
    $cmd = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe")
    )
    foreach ($p in $candidates) { if (Test-Path -LiteralPath $p) { return $p } }
    return $null
}

function Find-DockerDesktopExe {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe")
    )
    foreach ($p in $candidates) { if (Test-Path -LiteralPath $p) { return $p } }
    return $null
}

function Install-DockerDesktop {
    Write-Log "Docker Desktop not found; installing per-user WSL 2 edition"
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        $args = @("install", "--exact", "--id", "Docker.DockerDesktop", "--scope", "user", "--silent", "--accept-package-agreements", "--accept-source-agreements")
        $p = Start-Process -FilePath $winget.Source -ArgumentList $args -Wait -PassThru
        if ($p.ExitCode -eq 0 -and (Find-DockerExe)) { return }
    }

    $Installer = Join-Path $Cache "DockerDesktopInstaller.exe"
    Invoke-WebRequest -Uri $Lock.docker.installer_url -OutFile $Installer -UseBasicParsing
    $sig = Get-AuthenticodeSignature -LiteralPath $Installer
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "Docker") {
        throw "Docker Desktop installer signature is not valid"
    }
    $args = @("install", "--user", "--quiet", "--accept-license", "--backend=wsl-2", "--no-windows-containers")
    $p = Start-Process -FilePath $Installer -ArgumentList $args -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Docker Desktop installer failed with code $($p.ExitCode)" }
}

function Ensure-Docker {
    $docker = Find-DockerExe
    if (-not $docker) {
        Install-DockerDesktop
        $docker = Find-DockerExe
    }
    if (-not $docker) { throw "Docker CLI is unavailable after installation" }
    $dockerDir = Split-Path $docker
    if (($env:PATH -split ";") -notcontains $dockerDir) { $env:PATH = "$dockerDir;$env:PATH" }

    & $docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        $desktop = Find-DockerDesktopExe
        if ($desktop) { Start-Process -FilePath $desktop | Out-Null }
        $deadline = [DateTime]::UtcNow.AddMinutes(4)
        do {
            Start-Sleep -Seconds 3
            & $docker info *> $null
            if ($LASTEXITCODE -eq 0) { break }
        } while ([DateTime]::UtcNow -lt $deadline)
    }
    & $docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop did not become ready" }
    return $docker
}

function Test-Port([string]$HostName, [int]$Port) {
    try {
        $client = [Net.Sockets.TcpClient]::new()
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne(300)) { $client.Close(); return $false }
        $client.EndConnect($iar); $client.Close(); return $true
    } catch { return $false }
}

function Save-EnvironmentReceipt([string]$PythonExe, [string]$DockerExe) {
    $gpu = $null
    try { $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null) -join " | " } catch {}
    $dockerVersion = (& $DockerExe version --format "client={{.Client.Version}} server={{.Server.Version}}" 2>$null) -join " "
    $wslVersion = (& wsl.exe --version 2>$null) -join " | "
    $receipt = [ordered]@{
        time_utc = [DateTime]::UtcNow.ToString("o")
        bootstrap_lock_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $LockPath).Hash.ToLowerInvariant()
        python = (& $PythonExe --version 2>&1) -join " "
        docker = $dockerVersion
        wsl = $wslVersion
        gpu = $gpu
        root = $Root
    }
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Runtime "environment-receipt.json") -Encoding UTF8
}

if ($env:PROCESSOR_ARCHITECTURE -notin @("AMD64", "x86_64")) {
    throw "This bootstrap currently supports x64 Windows only"
}

Write-Log "NCPA Verifier bootstrap starting"
$Python = Ensure-PortablePython
Ensure-Wsl
$Docker = Ensure-Docker

Write-Log "Running verifier self-tests"
& $Python -m unittest discover -s (Join-Path $Root "tests") -q
if ($LASTEXITCODE -ne 0) { throw "Verifier self-tests failed" }
Save-EnvironmentReceipt $Python $Docker

$Config = Join-Path $Root ([string]$Lock.app.config)
$Baseline = Join-Path $Root ([string]$Lock.app.baseline)
$Workspace = Join-Path $Root ([string]$Lock.app.workspace)
$HostName = [string]$Lock.app.host
$Port = [int]$Lock.app.port

if (-not (Test-Port $HostName $Port)) {
    Write-Log "Starting verifier service"
    $args = @("-m", "verifier", "serve", "--config", $Config, "--baseline", $Baseline, "--workspace", $Workspace, "--host", $HostName, "--port", [string]$Port)
    $p = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $Runtime "server.pid") -Value $p.Id -Encoding ASCII
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do { Start-Sleep -Milliseconds 300 } while (-not (Test-Port $HostName $Port) -and [DateTime]::UtcNow -lt $deadline)
}

if (-not (Test-Port $HostName $Port)) { throw "Verifier service did not start" }
$url = "http://${HostName}:$Port/"
Write-Log "Verifier ready at $url"
Start-Process $url | Out-Null
