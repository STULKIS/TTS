# One-click, per-folder Windows CPU runtime. Never changes GSV, PATH or system Python.
# Downloads are confined to .voice-design beside the installed TTS folder.
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

# An extracted branch zip may be on USB. Offer the installed D:\TTS location
# before downloading models. No deletion/mirroring of the existing installation.
$Destination = if (Test-Path 'D:\') { 'D:\TTS' } else { Join-Path $env:SystemDrive 'TTS' }
if ($Repo.TrimEnd('\') -ine $Destination.TrimEnd('\')) {
    $Answer = Read-Host "Install/update in $Destination before downloading the model? [Y/n]"
    if ($Answer -notmatch '^[nN]') {
        & robocopy $Repo $Destination /E /XD .git .venv .voice-design node_modules /R:1 /W:1 /NFL /NDL /NJH /NJS
        if ($LASTEXITCODE -gt 7) { Write-Error 'Could not copy the Studio files to the install drive.'; exit 1 }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Destination 'tools\start_voice_design.ps1')
        exit $LASTEXITCODE
    }
}
$State = Join-Path $Repo '.voice-design'
$env:UV_CACHE_DIR = Join-Path $State 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $State 'python'
$env:HF_HOME = Join-Path $State 'huggingface'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_XET = '1'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE. See the error above. Re-run this launcher to retry." }
}

$SetupLock = $null
try {
    New-Item -ItemType Directory -Force $State | Out-Null
    $SetupLock = [IO.File]::Open((Join-Path $State 'session.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    Write-Host 'Gacha Voice Designer - experimental Qwen3 CPU mode'
    Write-Host 'Close Classic Studio before continuing. This model is larger and may be slow.'
    Write-Host 'No NVIDIA GPU, administrator access or manual Python install is required.'
    Write-Host "Runtime and model downloads stay here: $State"
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { throw 'This installer targets x64 Windows, not ARM64.' }
    # Do not accidentally start a second model in an already-running designer.
    if (Get-NetTCPConnection -LocalPort 7862 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Port 7862 is in use. If Voice Designer is already running, use its existing browser tab. Otherwise close the app using that port.'
    }
    $Ready = Join-Path $State 'ready.json'
    $Req = Join-Path $PSScriptRoot 'requirements-voice-design.txt'
    $Adapter = Join-Path $PSScriptRoot 'voice_design.py'
    $Fingerprint = (Get-FileHash $Req -Algorithm SHA256).Hash + (Get-FileHash $Adapter -Algorithm SHA256).Hash
    $Installed = $false
    if (Test-Path $Ready) {
        try { $Installed = ((Get-Content $Ready -Raw | ConvertFrom-Json).fingerprint -eq $Fingerprint) } catch { $Installed = $false }
    }
    $Python = Join-Path $State 'venv\Scripts\python.exe'
    if (-not (Test-Path $Python)) { $Installed = $false }
    if (-not $Installed) {
        $Drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($Repo).Substring(0,1))
        if ($Drive.Free -lt 16GB) { throw 'First setup needs at least 16 GB free on the TTS drive for runtime, model and caches.' }
        $Available = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
        if ($Available -lt 9GB) { throw 'At least 9 GB of RAM must currently be free. Close Classic Studio, games and large browser sessions, then try again.' }
        Write-Host 'First setup downloads several GB and runs a real model test. It can take a long time.'
        New-Item -ItemType Directory -Force $State | Out-Null
        $UvDir = Join-Path $State 'uv'
        $Uv = Join-Path $UvDir 'uv.exe'
        if (-not (Test-Path $Uv)) {
            $Zip = Join-Path $State 'uv.zip'
            $Url = 'https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-pc-windows-msvc.zip'
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
            $Expected = '515dc53d7553f1357d0abc1f70acd921fbb9e30230b1d9a08737236daa6ee920'
            if ((Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
                Remove-Item $Zip -Force
                throw 'Runtime bootstrap download failed SHA256 verification. No downloaded code was executed.'
            }
            Expand-Archive -Path $Zip -DestinationPath $UvDir -Force
            if (-not (Test-Path $Uv)) {
                $Found = Get-ChildItem -Path $UvDir -Recurse -Filter uv.exe | Select-Object -First 1
                if (-not $Found) { throw 'Verified bootstrap archive did not contain uv.exe.' }
                Copy-Item $Found.FullName $Uv
            }
            Remove-Item $Zip -Force
        }
        Invoke-Checked $Uv @('python', 'install', '3.12')
        if (-not (Test-Path $Python)) { Invoke-Checked $Uv @('venv', '--python', '3.12', (Join-Path $State 'venv')) }
        Invoke-Checked $Uv @('pip', 'install', '--python', $Python, '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.8.0', 'torchaudio==2.8.0')
        Invoke-Checked $Uv @('pip', 'install', '--python', $Python, '-r', $Req)
        Write-Host 'Checking the real voice model now. The first call also downloads the model.'
        Invoke-Checked $Python @($Adapter, '--self-test', (Join-Path $State 'setup-test.wav'))
        # Only real inference success marks setup ready. No stub can satisfy this.
        @{ fingerprint=$Fingerprint; model='Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'; tested=(Get-Date).ToString('o') } |
            ConvertTo-Json | Set-Content -Encoding UTF8 $Ready
        Write-Host 'Model test succeeded. setup-test.wav is available for listening in .voice-design.'
    }
    try {
        $Shell = New-Object -ComObject WScript.Shell
        $Desktop = [Environment]::GetFolderPath('Desktop')
        $Link = $Shell.CreateShortcut((Join-Path $Desktop 'Gacha Voice Designer.lnk'))
        $Link.TargetPath = Join-Path $Repo 'VOICE-DESIGN.bat'
        $Link.WorkingDirectory = $Repo
        $Link.Description = 'Gacha Voice Designer - model-driven traits and acting, CPU'
        $Link.IconLocation = 'shell32.dll,13'
        $Link.Save()
    } catch { Write-Warning 'Could not create the desktop icon; VOICE-DESIGN.bat still launches the app.' }
    Write-Host 'Opening Studio at http://127.0.0.1:7862. Keep this window open.'
    Invoke-Checked $Python @((Join-Path $PSScriptRoot 'type_ui.py'), '--engine', 'qwen-design', '--port', '7862', '--open')
} catch {
    Write-Host ''
    Write-Host ('Voice Designer could not start: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Your existing GPT-SoVITS installation was not modified.'
    exit 1
} finally {
    if ($null -ne $SetupLock) { $SetupLock.Dispose() }
}
