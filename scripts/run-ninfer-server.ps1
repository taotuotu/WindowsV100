[CmdletBinding()]
param(
    [string]$Model,
    [string]$Device,
    [int]$Context,
    [string]$KvDtype,
    [int]$DraftTokens,
    [int]$PrefillChunk,
    [string]$ListenHost,
    [int]$Port,
    [string]$ApiKey,
    [string]$RequestLog,
    [switch]$Stop,
    [string]$BuildDirectory = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArguments = @()
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$explicitNames = @{}
foreach ($name in $PSBoundParameters.Keys) { $explicitNames[$name] = $true }

$config = @{}
$configPath = Join-Path $repoRoot '.local\windows-server.psd1'
if (Test-Path -LiteralPath $configPath -PathType Leaf) {
    $loadedConfig = Import-PowerShellDataFile -LiteralPath $configPath
    if ($loadedConfig -isnot [System.Collections.IDictionary]) {
        throw "Local server configuration must be a PowerShell data hashtable: '$configPath'."
    }
    $config = $loadedConfig
}

function Get-EffectiveSetting {
    param(
        [string]$Name,
        [object]$ExplicitValue,
        [object]$DefaultValue
    )
    if ($explicitNames.ContainsKey($Name)) { return $ExplicitValue }
    if ($config.Contains($Name) -and $null -ne $config[$Name]) { return $config[$Name] }
    return $DefaultValue
}

$Model = [string](Get-EffectiveSetting 'Model' $Model '')
$Device = [string](Get-EffectiveSetting 'Device' $Device 'auto')
$Context = [int](Get-EffectiveSetting 'Context' $Context 8192)
$KvDtype = [string](Get-EffectiveSetting 'KvDtype' $KvDtype 'bf16')
$DraftTokens = [int](Get-EffectiveSetting 'DraftTokens' $DraftTokens 3)
$PrefillChunk = [int](Get-EffectiveSetting 'PrefillChunk' $PrefillChunk 512)
$ListenHost = [string](Get-EffectiveSetting 'ListenHost' $ListenHost '127.0.0.1')
$Port = [int](Get-EffectiveSetting 'Port' $Port 8110)
$ApiKey = [string](Get-EffectiveSetting 'ApiKey' $ApiKey '')
$RequestLog = [string](Get-EffectiveSetting 'RequestLog' $RequestLog '')

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $repoRoot 'models\qwen3_8_27b_nvfp4_v2.ninfer'
} elseif (-not [System.IO.Path]::IsPathRooted($Model)) {
    $modelBase = if ($explicitNames.ContainsKey('Model')) { (Get-Location).Path } else { $repoRoot }
    $Model = [System.IO.Path]::GetFullPath((Join-Path $modelBase $Model))
}

if ($Device -ne 'auto' -and $Device -notmatch '^\d+$') {
    throw "Device must be 'auto' or a zero-based CUDA device index."
}
if ($Context -lt 1 -or $Context -gt 1048576) {
    throw 'Context must be between 1 and 1048576 tokens.'
}
if ($KvDtype -notin @('bf16', 'int8', 'fp8')) {
    throw 'KvDtype must be bf16, int8, or fp8 for the Windows V100 server.'
}
if ($DraftTokens -lt 1 -or $DraftTokens -gt 7) {
    throw 'DraftTokens must be between 1 and 7 for MTP.'
}
if ($PrefillChunk -lt 128 -or $PrefillChunk -gt 65536 -or ($PrefillChunk % 128) -ne 0) {
    throw 'PrefillChunk must be a multiple of 128 between 128 and 65536.'
}
if ([string]::IsNullOrWhiteSpace($ListenHost)) { throw 'ListenHost must not be empty.' }
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }

if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $repoRoot 'build-win-v100'
} elseif (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)

$serverCandidates = @()
if ($explicitNames.ContainsKey('BuildDirectory')) {
    $serverCandidates += Join-Path $BuildDirectory 'apps\windows-serve\Release\ninfer-windows-serve.exe'
}
$serverCandidates += @(
    (Join-Path $repoRoot 'build-win-v100\apps\windows-serve\Release\ninfer-windows-serve.exe'),
    (Join-Path $BuildDirectory 'apps\windows-serve\Release\ninfer-windows-serve.exe'),
    (Join-Path $repoRoot 'bin\ninfer-windows-serve.exe'),
    (Join-Path $repoRoot 'dist\bin\ninfer-windows-serve.exe')
)
$serverCandidates = $serverCandidates | Select-Object -Unique
$serverExe = $serverCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $serverExe) {
    throw 'NInfer server executable was not found. Build it with scripts\build-windows-v100.ps1 -Step server or extract a release package.'
}
$serverExe = (Resolve-Path -LiteralPath $serverExe).Path

$listenArgument = $ListenHost.Trim([char[]]@('[', ']'))
$probeHost = $listenArgument
if ($probeHost -eq '0.0.0.0') { $probeHost = '127.0.0.1' }
if ($probeHost -eq '::') { $probeHost = '::1' }
if ($probeHost -eq 'localhost') { $probeHost = '127.0.0.1' }
$uriHost = $probeHost
if ($uriHost.Contains(':')) { $uriHost = '[' + $uriHost + ']' }
$baseUri = 'http://' + $uriHost + ':' + $Port

function Get-SelectedListenAddresses {
    param([string]$HostName)
    $addresses = [System.Collections.Generic.List[string]]::new()
    if ($HostName -eq 'localhost') {
        $addresses.Add('127.0.0.1')
        $addresses.Add('::1')
        $addresses.Add('0.0.0.0')
        $addresses.Add('::')
    } elseif ($HostName -eq '0.0.0.0' -or $HostName -eq '::') {
        $addresses.Add($HostName)
    } else {
        $parsedAddress = $null
        if ([System.Net.IPAddress]::TryParse($HostName, [ref]$parsedAddress)) {
            $addresses.Add($parsedAddress.ToString())
            if ($parsedAddress.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork) {
                $addresses.Add('0.0.0.0')
            } else {
                $addresses.Add('::')
            }
        } else {
            foreach ($address in [System.Net.Dns]::GetHostAddresses($HostName)) {
                $addresses.Add($address.ToString())
                if ($address.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork) {
                    $addresses.Add('0.0.0.0')
                } else {
                    $addresses.Add('::')
                }
            }
        }
    }
    return @($addresses | Sort-Object -Unique)
}

function Get-ListenersOnPort {
    param([int]$LocalPort)
    try {
        return @(Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction Stop)
    } catch {
        if (-not (Get-Command netstat.exe -ErrorAction SilentlyContinue)) {
            throw ('Unable to inspect local listeners on port {0}: {1}' -f $LocalPort, $_.Exception.Message)
        }
        $listeners = @()
        foreach ($line in (& netstat.exe -ano -p tcp)) {
            if ($line -match '^\s*TCP\s+(\S+):(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$' -and
                [int]$Matches[2] -eq $LocalPort) {
                $listeners += [pscustomobject]@{
                    LocalAddress = $Matches[1].Trim([char[]]@('[', ']'))
                    OwningProcess = [int]$Matches[3]
                }
            }
        }
        return $listeners
    }
}

if ($Stop) {
    $selectedAddresses = Get-SelectedListenAddresses $listenArgument
    $listeners = Get-ListenersOnPort $Port
    $matchingListeners = @($listeners | Where-Object {
        $selectedAddresses -contains ([string]$_.LocalAddress).Trim([char[]]@('[', ']'))
    })
    $listenerIds = @($matchingListeners |
        ForEach-Object { [int]$_.OwningProcess } |
        Sort-Object -Unique)
    if ($listenerIds.Count -eq 0) {
        Write-Host ('No listener for {0}:{1} was found.' -f $ListenHost, $Port)
        return
    }
    if ($listenerIds.Count -ne 1) {
        throw ('Found multiple listener processes for {0}:{1}; no process was stopped.' -f $ListenHost, $Port)
    }

    $listenerId = $listenerIds[0]
    try {
        $listenerProcess = Get-Process -Id $listenerId -ErrorAction Stop
        if (-not $listenerProcess.Path) { throw 'process path unavailable' }
        $listenerPath = [System.IO.Path]::GetFullPath($listenerProcess.Path)
    } catch {
        throw "Cannot verify listener PID $listenerId executable path; no process was stopped."
    }
    if ($listenerPath -ine $serverExe) {
        throw ('PID {0} on {1}:{2} is {3}, not {4}. No process was stopped.' -f
            $listenerId, $ListenHost, $Port, $listenerPath, $serverExe)
    }

    Stop-Process -Id $listenerId -ErrorAction Stop
    Write-Host ('Stopped NInfer server PID {0} on {1}:{2}.' -f $listenerId, $ListenHost, $Port)
    return
}

# Avoid loading a second copy of the model when this port already serves the expected API.
$modelAlias = 'qwen3.8-27b'
$healthOk = $false
$existingIds = @()
try {
    $health = Invoke-RestMethod -Uri "$baseUri/health" -TimeoutSec 2
    $healthOk = $health.status -eq 'ok'
    if ($healthOk) {
        if ($ApiKey) {
            $models = Invoke-RestMethod -Uri "$baseUri/v1/models" -Headers @{ Authorization = "Bearer $ApiKey" } -TimeoutSec 2
        } else {
            $models = Invoke-RestMethod -Uri "$baseUri/v1/models" -TimeoutSec 2
        }
        $existingIds = @($models.data | ForEach-Object { [string]$_.id })
    }
} catch {
    # A protected or unrelated service is still detected by the TCP check below.
}

$portOpen = $false
$client = [System.Net.Sockets.TcpClient]::new()
try {
    $connect = $client.BeginConnect($probeHost, $Port, $null, $null)
    if ($connect.AsyncWaitHandle.WaitOne(400) -and $client.Connected) {
        $client.EndConnect($connect)
        $portOpen = $true
    }
} catch {
    $portOpen = $false
} finally {
    $client.Close()
}
if ($portOpen) {
    if ($healthOk -and $existingIds -contains $modelAlias) {
        Write-Host "NInfer is already serving $modelAlias at $baseUri. Reusing the running service; this launch's model/device/context settings were not applied."
        return
    }
    $reported = if ($existingIds.Count -gt 0) { $existingIds -join ', ' } else { 'model identity unavailable' }
    throw "Port $Port on $probeHost is already in use ($reported). No process was stopped."
}

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model file was not found at '$Model'. Download the pinned V2 model with scripts\download-model.ps1 or pass -Model <path>."
}

$runtimeDirectories = [System.Collections.Generic.List[string]]::new()
$releaseDll = Join-Path $repoRoot 'bin\cudart64_12.dll'
if (Test-Path -LiteralPath $releaseDll -PathType Leaf) {
    $runtimeDirectories.Add((Split-Path -Parent $releaseDll))
}
$exeDll = Join-Path (Split-Path -Parent $serverExe) 'cudart64_12.dll'
if (Test-Path -LiteralPath $exeDll -PathType Leaf) {
    $runtimeDirectories.Add((Split-Path -Parent $exeDll))
}
if ($env:CUDA_PATH) { $runtimeDirectories.Add((Join-Path $env:CUDA_PATH 'bin')) }
Get-ChildItem Env: | Where-Object { $_.Name -match '^CUDA_PATH_V12_' } |
    ForEach-Object { $runtimeDirectories.Add((Join-Path $_.Value 'bin')) }
if ($env:ProgramFiles) {
    $cudaInstallRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $cudaInstallRoot -PathType Container) {
        Get-ChildItem -LiteralPath $cudaInstallRoot -Directory |
            Where-Object { $_.Name -match '^v12\.\d+$' } |
            Sort-Object { [version]($_.Name.Substring(1)) } -Descending |
            ForEach-Object { $runtimeDirectories.Add((Join-Path $_.FullName 'bin')) }
    }
}
$runtimeDirectories.AddRange([string[]]($env:PATH -split ';' | Where-Object { $_ }))
$runtimeDirectory = $runtimeDirectories |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ 'cudart64_12.dll') -PathType Leaf } |
    Select-Object -First 1
if (-not $runtimeDirectory) {
    throw 'cudart64_12.dll was not found beside the release binary, in CUDA_PATH, or in an installed CUDA 12 toolkit.'
}
if (-not (($env:PATH -split ';') -contains $runtimeDirectory) -and
    (Split-Path -Parent $serverExe) -ine $runtimeDirectory) {
    $env:PATH = "$runtimeDirectory;$env:PATH"
}

$serverArgs = @('--model', $Model)
$serverArgs += @(
    '--device', $Device,
    '--max-context', "$Context",
    '--kv-capacity', "$Context",
    '--kv-dtype', $KvDtype,
    '--draft-tokens', "$DraftTokens",
    '--prefill-chunk', "$PrefillChunk"
)
if ($ApiKey) { $serverArgs += @('--api-key', $ApiKey) }
if (-not [string]::IsNullOrWhiteSpace($RequestLog)) {
    if (-not [System.IO.Path]::IsPathRooted($RequestLog)) {
        $RequestLog = Join-Path $repoRoot $RequestLog
    }
    $RequestLog = [System.IO.Path]::GetFullPath($RequestLog)
    $requestLogDirectory = Split-Path -Parent $RequestLog
    if ($requestLogDirectory -and -not (Test-Path -LiteralPath $requestLogDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $requestLogDirectory -Force | Out-Null
    }
    $serverArgs += @('--request-log-jsonl', $RequestLog)
}
$serverArgs += @('--host', $listenArgument, '--port', "$Port", '--cors')
if ($ServerArguments) { $serverArgs += $ServerArguments }

Write-Host "Starting NInfer at $baseUri with device $Device and context $Context. Press Ctrl+C to stop."
Push-Location $repoRoot
try {
    & $serverExe @serverArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NInfer server exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
