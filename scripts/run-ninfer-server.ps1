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
    [switch]$Vision,
    [string]$RequestLog,
    [switch]$Stop,
    [switch]$OpenChat,
    [ValidateRange(1, 86400)]
    [int]$ReadyTimeoutSeconds = 600,
    [string]$BuildDirectory = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArguments = @()
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($Stop) {
    if ($OpenChat) { throw '-OpenChat cannot be combined with -Stop.' }
    & (Join-Path $PSScriptRoot 'stop-ninfer.ps1')
    return
}
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
$Context = [int](Get-EffectiveSetting 'Context' $Context 143600)
$KvDtype = [string](Get-EffectiveSetting 'KvDtype' $KvDtype 'bf16')
$DraftTokens = [int](Get-EffectiveSetting 'DraftTokens' $DraftTokens 6)
$PrefillChunk = [int](Get-EffectiveSetting 'PrefillChunk' $PrefillChunk 2048)
$ListenHost = [string](Get-EffectiveSetting 'ListenHost' $ListenHost '127.0.0.1')
$Port = [int](Get-EffectiveSetting 'Port' $Port 8110)
$ApiKey = [string](Get-EffectiveSetting 'ApiKey' $ApiKey '')
$visionValue = Get-EffectiveSetting 'Vision' $Vision $false
if ($visionValue -is [System.Management.Automation.SwitchParameter]) {
    $VisionEnabled = $visionValue.IsPresent
} elseif ($visionValue -is [bool]) {
    $VisionEnabled = $visionValue
} else {
    throw 'Vision in .local\windows-server.psd1 must be $true or $false; use -Vision to enable it for one launch.'
}
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
$chatUrl = $baseUri + '/'

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

function Get-NInferProcessesForPort {
    param(
        [string]$ExecutablePath,
        [int]$LocalPort
    )

    try {
        $records = @(Get-CimInstance -ClassName Win32_Process -Filter "Name='$([System.IO.Path]::GetFileName($ExecutablePath))'" -ErrorAction Stop)
    } catch {
        throw ('Cannot inspect running NInfer processes safely: {0}' -f $_.Exception.Message)
    }

    $matching = @()
    foreach ($record in $records) {
        $commandLine = [string]$record.CommandLine
        $portMatch = [regex]::Match($commandLine, '(?:^|\s)"?--port"?\s+(?:"(?<quoted>\d+)"|(?<plain>\d+))(?=\s|$)')
        if ($portMatch.Success) {
            $processPort = if ($portMatch.Groups['quoted'].Success) { $portMatch.Groups['quoted'].Value } else { $portMatch.Groups['plain'].Value }
            if ($processPort -ne [string]$LocalPort) { continue }
        } elseif ($commandLine -match '(?:^|\s)"?--port"?(?:\s|$)' -or $LocalPort -ne 8110) {
            continue
        }

        if ([string]::IsNullOrWhiteSpace([string]$record.ExecutablePath)) {
            throw ('Cannot verify executable path for PID {0} using port {1}; no second server was started.' -f $record.ProcessId, $LocalPort)
        }
        $recordPath = [System.IO.Path]::GetFullPath([string]$record.ExecutablePath)
        if ($recordPath -ieq $ExecutablePath) { $matching += $record }
    }
    return $matching
}

function Get-HttpStatusCodeFromException {
    param([System.Exception]$Exception)
    $current = $Exception
    while ($null -ne $current) {
        try {
            if ($null -ne $current.Response -and $null -ne $current.Response.StatusCode) {
                return [int]$current.Response.StatusCode
            }
        } catch { }
        try {
            if ($null -ne $current.StatusCode) { return [int]$current.StatusCode }
        } catch { }
        $current = $current.InnerException
    }
    return $null
}

function Get-NInferServiceProbe {
    param(
        [string]$ServiceUri,
        [string]$ProbeApiKey
    )

    $result = [pscustomobject]@{
        HealthOk = $false
        ModelListAvailable = $false
        ModelIds = @()
        ModelsAuthStatusCode = $null
        VisionEnabled = $null
    }
    try {
        $health = Invoke-RestMethod -Uri "$ServiceUri/health" -TimeoutSec 2
    } catch {
        return $result
    }
    $result.HealthOk = $health.status -eq 'ok'
    if (-not $result.HealthOk) { return $result }

    try {
        if ($ProbeApiKey) {
            $models = Invoke-RestMethod -Uri "$ServiceUri/v1/models" -Headers @{ Authorization = "Bearer $ProbeApiKey" } -TimeoutSec 2
        } else {
            $models = Invoke-RestMethod -Uri "$ServiceUri/v1/models" -TimeoutSec 2
        }
        $result.ModelListAvailable = $true
        $result.ModelIds = @($models.data | ForEach-Object { [string]$_.id })
    } catch {
        $statusCode = Get-HttpStatusCodeFromException -Exception $_.Exception
        if ($statusCode -in @(401, 403)) { $result.ModelsAuthStatusCode = $statusCode }
    }

    try {
        if ($ProbeApiKey) {
            $modelInfo = Invoke-RestMethod -Uri "$ServiceUri/ui/model-info" -Headers @{ Authorization = "Bearer $ProbeApiKey" } -TimeoutSec 2
        } else {
            $modelInfo = Invoke-RestMethod -Uri "$ServiceUri/ui/model-info" -TimeoutSec 2
        }
        $visionProperty = $modelInfo.PSObject.Properties['vision_enabled']
        if ($null -ne $visionProperty -and $visionProperty.Value -is [bool]) {
            $result.VisionEnabled = $visionProperty.Value
        }
    } catch {
        # Older services and transient model-info failures leave Vision readiness unknown.
    }
    return $result
}

function Assert-NInferListenerExecutable {
    param(
        [string]$HostName,
        [int]$LocalPort,
        [string]$ExpectedExecutable
    )

    $selectedAddresses = @(Get-SelectedListenAddresses $HostName)
    $listeners = @(Get-ListenersOnPort $LocalPort | Where-Object {
        $selectedAddresses -contains ([string]$_.LocalAddress).Trim([char[]]@('[', ']'))
    })
    if ($listeners.Count -eq 0) {
        throw ('The API at port {0} responded, but no listener on the configured address could be verified as {1}.' -f $LocalPort, $ExpectedExecutable)
    }

    foreach ($listener in $listeners) {
        $listenerId = [int]$listener.OwningProcess
        try {
            $listenerProcess = Get-Process -Id $listenerId -ErrorAction Stop
            if (-not $listenerProcess.Path) { throw 'process path unavailable' }
            $listenerPath = [System.IO.Path]::GetFullPath($listenerProcess.Path)
        } catch {
            throw ('Cannot verify listener PID {0} on port {1}; refusing to reuse the API.' -f $listenerId, $LocalPort)
        }
        if ($listenerPath -ine $ExpectedExecutable) {
            throw ('Port {0} is serving an API from PID {1} at {2}, not the selected NInfer executable {3}; refusing to reuse it.' -f
                $LocalPort, $listenerId, $listenerPath, $ExpectedExecutable)
        }
    }
}

function Get-ModelsAuthorizationError {
    param(
        [string]$ConfiguredApiKey,
        [int]$StatusCode
    )
    if ($ConfiguredApiKey) {
        return ('GET /v1/models rejected the configured API key with HTTP {0}. Check the key in .local\windows-server.psd1 or pass -ApiKey.' -f $StatusCode)
    }
    return ('GET /v1/models returned HTTP {0}, and no API key is configured. Set ApiKey in .local\windows-server.psd1 or pass -ApiKey.' -f $StatusCode)
}

# Avoid loading a second copy of the model when this port already serves the expected API.
$modelAlias = 'qwen3.8-27b'
if ($OpenChat -and $ServerArguments) {
    for ($index = 0; $index -lt $ServerArguments.Count; $index++) {
        $argument = [string]$ServerArguments[$index]
        if ($argument -eq '--vision') {
            throw 'In -OpenChat mode, use the launcher -Vision switch instead of a raw --vision server argument.'
        }
        if ($argument -match '^(--model-id)=(.*)$') {
            throw 'In -OpenChat mode, pass --model-id and its value as separate server arguments.'
        }
        if ($argument -in @('--model', '-m', '--host', '--port', '--api-key') -or
            $argument -match '^(--model|--host|--port|--api-key)=') {
            throw ('In -OpenChat mode, use the named launcher parameter for {0} so readiness checks match the running service.' -f $argument.Split('=')[0])
        }
        if ($argument -eq '--model-id') {
            if (($index + 1) -ge $ServerArguments.Count -or
                [string]::IsNullOrWhiteSpace([string]$ServerArguments[$index + 1]) -or
                [string]$ServerArguments[$index + 1] -match '^-') {
                throw '--model-id in -OpenChat mode requires a non-empty value that does not begin with a dash.'
            }
            $modelAlias = [string]$ServerArguments[$index + 1]
            $index++
        }
    }
}
$healthOk = $false
$existingIds = @()
$initialProbe = Get-NInferServiceProbe -ServiceUri $baseUri -ProbeApiKey $ApiKey
$healthOk = $initialProbe.HealthOk
$existingIds = @($initialProbe.ModelIds)

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
        if ($OpenChat) {
            Assert-NInferListenerExecutable -HostName $listenArgument -LocalPort $Port -ExpectedExecutable $serverExe
        }
        if ($VisionEnabled -and $initialProbe.VisionEnabled -ne $true) {
            $visionState = if ($null -eq $initialProbe.VisionEnabled) { 'missing or unavailable' } else { 'disabled' }
            throw ('The healthy NInfer service at {0} serves {1}, but vision_enabled is {2}. It was left running. Run stop-ninfer.bat, then start-ninfer.bat -Vision.' -f
                $baseUri, $modelAlias, $visionState)
        }
        if ($OpenChat) {
            Write-Host "NInfer is already serving $modelAlias at $baseUri. Reusing the running service; this launch's model/device/context settings were not applied."
            try {
                Start-Process -FilePath $chatUrl -WindowStyle Normal -ErrorAction Stop | Out-Null
            } catch {
                Write-Warning ('The chat page could not be opened automatically: {0}' -f $_.Exception.Message)
                Write-Host ('Open or copy this URL: {0}' -f $chatUrl)
            }
            return
        }
        Write-Host "NInfer is already serving $modelAlias at $baseUri. Reusing the running service; this launch's model/device/context settings were not applied."
        return
    }
    if (-not $OpenChat) {
        $reported = if ($existingIds.Count -gt 0) { $existingIds -join ', ' } else { 'model identity unavailable' }
        throw "Port $Port on $probeHost is already in use ($reported). No process was stopped."
    }
}
if ($OpenChat -and $initialProbe.ModelsAuthStatusCode -in @(401, 403)) {
    Assert-NInferListenerExecutable -HostName $listenArgument -LocalPort $Port -ExpectedExecutable $serverExe
    throw (Get-ModelsAuthorizationError -ConfiguredApiKey $ApiKey -StatusCode $initialProbe.ModelsAuthStatusCode)
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
if ($VisionEnabled) { $serverArgs += '--vision' }
if ($ServerArguments) { $serverArgs += $ServerArguments }

if ($OpenChat) {
    Write-Host "Checking NInfer readiness at $baseUri. Ctrl+C stops only a server started by this launcher."
} else {
    Write-Host "Starting NInfer at $baseUri with device $Device and context $Context. Press Ctrl+C to stop."
}
Push-Location $repoRoot
try {
    if ($OpenChat) {
        $mutexMaterial = $serverExe.ToUpperInvariant() + '|' + $Port
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $mutexHash = [BitConverter]::ToString($sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($mutexMaterial))).Replace('-', '')
        } finally {
            $sha256.Dispose()
        }
        $launcherMutex = [System.Threading.Mutex]::new($false, ('Local\NInferLauncher-' + $mutexHash))
        $mutexOwned = $false
        $startedServerProcess = $null
        $readyDeadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
        try {
            while ($true) {
                # Always check readiness before the mutex. Another launcher may own the mutex
                # while its native server is loading; a ready service can be reused immediately.
                $probe = Get-NInferServiceProbe -ServiceUri $baseUri -ProbeApiKey $ApiKey
                $probeHealthy = $probe.HealthOk
                $probeIds = @($probe.ModelIds)
                if ($probe.ModelsAuthStatusCode -in @(401, 403)) {
                    Assert-NInferListenerExecutable -HostName $listenArgument -LocalPort $Port -ExpectedExecutable $serverExe
                    throw (Get-ModelsAuthorizationError -ConfiguredApiKey $ApiKey -StatusCode $probe.ModelsAuthStatusCode)
                }
                $healthyExpectedModel = $probeHealthy -and ($probeIds -contains $modelAlias)
                if ($healthyExpectedModel -and $VisionEnabled -and $probe.VisionEnabled -ne $true) {
                    Assert-NInferListenerExecutable -HostName $listenArgument -LocalPort $Port -ExpectedExecutable $serverExe
                    if ($probe.VisionEnabled -eq $false -and $startedServerProcess) {
                        throw ('This launch started NInfer with Vision requested, but /ui/model-info reports vision_enabled=false at {0}.' -f $baseUri)
                    }
                    if (-not $startedServerProcess) {
                        $visionState = if ($null -eq $probe.VisionEnabled) { 'missing or unavailable' } else { 'disabled' }
                        throw ('The healthy NInfer service at {0} serves {1}, but vision_enabled is {2}. It was left running. Run stop-ninfer.bat, then start-ninfer.bat -Vision.' -f
                            $baseUri, $modelAlias, $visionState)
                    }
                }
                if ($healthyExpectedModel -and $VisionEnabled -and
                    $null -eq $probe.VisionEnabled -and $startedServerProcess) {
                    if ([DateTime]::UtcNow -ge $readyDeadline) {
                        throw ('Timed out after {0} seconds waiting for vision_enabled=true at {1}.' -f
                            $ReadyTimeoutSeconds, $baseUri)
                    }
                    if ($startedServerProcess.HasExited) {
                        throw "NInfer server exited with code $($startedServerProcess.ExitCode) before reporting Vision readiness."
                    }
                    Start-Sleep -Milliseconds 1000
                    continue
                }
                $ready = $healthyExpectedModel -and (-not $VisionEnabled -or $probe.VisionEnabled -eq $true)
                if ($ready) {
                    Assert-NInferListenerExecutable -HostName $listenArgument -LocalPort $Port -ExpectedExecutable $serverExe
                    Write-Host "NInfer is ready with model $modelAlias at $baseUri."
                    try {
                        Start-Process -FilePath $chatUrl -WindowStyle Normal -ErrorAction Stop | Out-Null
                    } catch {
                        Write-Warning ('The chat page could not be opened automatically: {0}' -f $_.Exception.Message)
                        Write-Host ('Open or copy this URL: {0}' -f $chatUrl)
                    }
                    if ($startedServerProcess) {
                        $startedServerProcess.WaitForExit()
                        if ($startedServerProcess.ExitCode -ne 0) {
                            throw "NInfer server exited with code $($startedServerProcess.ExitCode)."
                        }
                    }
                    return
                }

                $portListeners = @(Get-ListenersOnPort $Port | Where-Object {
                    (Get-SelectedListenAddresses $listenArgument) -contains ([string]$_.LocalAddress).Trim([char[]]@('[', ']'))
                })
                $matchingProcesses = @(Get-NInferProcessesForPort -ExecutablePath $serverExe -LocalPort $Port)

                $matchingListenerPids = @()
                foreach ($listener in $portListeners) {
                    $listenerId = [int]$listener.OwningProcess
                    try {
                        $listenerProcess = Get-Process -Id $listenerId -ErrorAction Stop
                        if (-not $listenerProcess.Path) { throw 'process path unavailable' }
                        $listenerPath = [System.IO.Path]::GetFullPath($listenerProcess.Path)
                    } catch {
                        throw ('Cannot verify PID {0} on port {1}; no process was stopped or started.' -f $listenerId, $Port)
                    }
                    if ($listenerPath -ieq $serverExe) {
                        $matchingListenerPids += $listenerId
                    } else {
                        throw ('Port {0} on {1} is occupied by PID {2} ({3}), not the NInfer server executable. No process was stopped or started.' -f
                            $Port, $probeHost, $listenerId, $listenerPath)
                    }
                }

                $tcpOpen = $false
                $probeClient = [System.Net.Sockets.TcpClient]::new()
                try {
                    $connect = $probeClient.BeginConnect($probeHost, $Port, $null, $null)
                    if ($connect.AsyncWaitHandle.WaitOne(400) -and $probeClient.Connected) {
                        $probeClient.EndConnect($connect)
                        $tcpOpen = $true
                    }
                } catch {
                    $tcpOpen = $false
                } finally {
                    $probeClient.Close()
                }
                if ($tcpOpen -and $matchingListenerPids.Count -eq 0) {
                    throw ('Port {0} on {1} is occupied, but its listener could not be verified as this NInfer server. No process was stopped or started.' -f $Port, $probeHost)
                }

                $sameServerPresent = ($matchingProcesses.Count -gt 0 -or $matchingListenerPids.Count -gt 0)
                if ($probeHealthy -and $probeIds.Count -gt 0 -and $sameServerPresent -and
                    $probeIds -notcontains $modelAlias) {
                    throw ('NInfer is healthy at {0}, but serves [{1}] instead of expected alias {2}. No process was stopped.' -f
                        $baseUri, ($probeIds -join ', '), $modelAlias)
                }

                if (-not $sameServerPresent) {
                    if (-not $mutexOwned) {
                        try { $mutexOwned = $launcherMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $mutexOwned = $true }
                    }
                    if ($mutexOwned) {
                        # Recheck after acquiring the lock so simultaneous launchers cannot both start a model.
                        $lockProbe = Get-NInferServiceProbe -ServiceUri $baseUri -ProbeApiKey $ApiKey
                        $probeAfterLock = $lockProbe.HealthOk -and (@($lockProbe.ModelIds) -contains $modelAlias)
                        if (-not $probeAfterLock) {
                            $nativeListenerPresent = $false
                            $lockedListeners = @(Get-ListenersOnPort $Port | Where-Object {
                                (Get-SelectedListenAddresses $listenArgument) -contains ([string]$_.LocalAddress).Trim([char[]]@('[', ']'))
                            })
                            foreach ($listener in $lockedListeners) {
                                $listenerId = [int]$listener.OwningProcess
                                try {
                                    $listenerProcess = Get-Process -Id $listenerId -ErrorAction Stop
                                    if (-not $listenerProcess.Path) { throw 'process path unavailable' }
                                    $listenerPath = [System.IO.Path]::GetFullPath($listenerProcess.Path)
                                } catch {
                                    throw ('Cannot verify PID {0} on port {1}; no process was stopped or started.' -f $listenerId, $Port)
                                }
                                if ($listenerPath -ine $serverExe) {
                                    throw ('Port {0} on {1} is occupied by PID {2} ({3}), not the NInfer server executable. No process was stopped or started.' -f
                                        $Port, $probeHost, $listenerId, $listenerPath)
                                }
                                $nativeListenerPresent = $true
                            }
                            $lockedTcpOpen = $false
                            $lockedClient = [System.Net.Sockets.TcpClient]::new()
                            try {
                                $lockedConnect = $lockedClient.BeginConnect($probeHost, $Port, $null, $null)
                                if ($lockedConnect.AsyncWaitHandle.WaitOne(400) -and $lockedClient.Connected) {
                                    $lockedClient.EndConnect($lockedConnect)
                                    $lockedTcpOpen = $true
                                }
                            } catch {
                                $lockedTcpOpen = $false
                            } finally {
                                $lockedClient.Close()
                            }
                            if ($lockedTcpOpen -and -not $nativeListenerPresent) {
                                throw ('Port {0} on {1} is occupied, but its listener could not be verified as this NInfer server. No process was stopped or started.' -f $Port, $probeHost)
                            }
                            $stillRunning = @(Get-NInferProcessesForPort -ExecutablePath $serverExe -LocalPort $Port)
                            if ($stillRunning.Count -eq 0 -and -not $nativeListenerPresent) {
                                $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
                                $startInfo.FileName = $serverExe
                                $startInfo.WorkingDirectory = $repoRoot
                                $startInfo.UseShellExecute = $false
                                $startInfo.CreateNoWindow = $false
                                $startInfo.RedirectStandardOutput = $false
                                $startInfo.RedirectStandardError = $false
                                $quotedArguments = foreach ($argument in $serverArgs) {
                                    $value = [string]$argument
                                    $builder = [System.Text.StringBuilder]::new()
                                    [void]$builder.Append('"')
                                    $slashes = 0
                                    foreach ($character in $value.ToCharArray()) {
                                        if ($character -eq [char]92) {
                                            $slashes++
                                        } elseif ($character -eq [char]34) {
                                            [void]$builder.Append([string]::new([char]92, (2 * $slashes + 1)))
                                            [void]$builder.Append('"')
                                            $slashes = 0
                                        } else {
                                            if ($slashes -gt 0) { [void]$builder.Append([string]::new([char]92, $slashes)) }
                                            [void]$builder.Append($character)
                                            $slashes = 0
                                        }
                                    }
                                    if ($slashes -gt 0) { [void]$builder.Append([string]::new([char]92, (2 * $slashes))) }
                                    [void]$builder.Append('"')
                                    $builder.ToString()
                                }
                                $startInfo.Arguments = [string]::Join(' ', [string[]]$quotedArguments)
                                $startedServerProcess = [System.Diagnostics.Process]::new()
                                $startedServerProcess.StartInfo = $startInfo
                                if (-not $startedServerProcess.Start()) { throw 'Could not start the NInfer server process.' }
                                Write-Host ('Started NInfer server PID {0}; waiting up to {1} seconds for /health and model alias {2}.' -f
                                    $startedServerProcess.Id, $ReadyTimeoutSeconds, $modelAlias)
                            }
                        } else {
                            # The service became ready during the lock handoff; the next pass opens chat.
                            $mutexOwned = $true
                        }
                    }
                }

                if ([DateTime]::UtcNow -ge $readyDeadline) {
                    $readinessRequirement = if ($VisionEnabled) {
                        'health, model alias, and vision_enabled=true'
                    } else {
                        'health and model alias'
                    }
                    throw ('Timed out after {0} seconds waiting for NInfer {1} at {2}.' -f
                        $ReadyTimeoutSeconds, $readinessRequirement, $baseUri)
                }
                if ($startedServerProcess -and $startedServerProcess.HasExited) {
                    throw "NInfer server exited with code $($startedServerProcess.ExitCode) before becoming ready."
                }
                Start-Sleep -Milliseconds 1000
            }
        } finally {
            if ($startedServerProcess) {
                try {
                    if (-not $startedServerProcess.HasExited) {
                        $startedServerProcess.Kill()
                        $startedServerProcess.WaitForExit(5000) | Out-Null
                    }
                } catch {
                    Write-Warning ('Could not stop launcher-owned NInfer PID {0}: {1}' -f $startedServerProcess.Id, $_.Exception.Message)
                }
                $startedServerProcess.Dispose()
            }
            if ($mutexOwned) { $launcherMutex.ReleaseMutex() }
            $launcherMutex.Dispose()
        }
    }
    & $serverExe @serverArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NInfer server exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
