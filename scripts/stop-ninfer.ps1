[CmdletBinding()]
param(
    [switch]$List,
    [string]$ApiKey
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot '..')).Path)
if (-not $PSBoundParameters.ContainsKey('ApiKey')) {
    $taskConfigPath = Join-Path $repoRoot '.local/windows-server.psd1'
    if (Test-Path -LiteralPath $taskConfigPath -PathType Leaf) {
        try {
            $taskConfig = Import-PowerShellDataFile -LiteralPath $taskConfigPath
            if ($taskConfig.ContainsKey('ApiKey')) { $ApiKey = [string]$taskConfig.ApiKey }
        } catch {
            Write-Warning 'Personal configuration could not be read; authenticated shutdown may require -ApiKey.'
        }
    }
}
$executableNames = @(
    'ninfer-windows-serve.exe',
    'ninfer-windows-text.exe',
    'ninfer.exe',
    'ninfer-serve.exe',
    'ninfer-perplexity.exe'
)

function ConvertTo-NormalizedPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\', '/'))
}

function Test-PathWithinDirectory {
    param(
        [string]$Path,
        [string]$Directory
    )
    try {
        $fullPath = ConvertTo-NormalizedPath $Path
        $fullDirectory = ConvertTo-NormalizedPath $Directory
    } catch {
        return $false
    }
    $prefix = $fullDirectory + [System.IO.Path]::DirectorySeparatorChar
    return $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

$allowedDirectories = [System.Collections.Generic.List[string]]::new()
$allowedDirectories.Add((Join-Path $repoRoot 'bin'))
$allowedDirectories.Add((Join-Path $repoRoot 'dist'))
$allowedDirectories.Add((Join-Path $repoRoot '.local'))
Get-ChildItem -LiteralPath $repoRoot -Directory -ErrorAction Stop |
    Where-Object { $_.Name -match '^build($|[-_])' } |
    ForEach-Object { $allowedDirectories.Add($_.FullName) }

function Test-IsRepositoryNInferProcess {
    param(
        [string]$Name,
        [string]$ExecutablePath
    )
    if ($executableNames -notcontains $Name -or [string]::IsNullOrWhiteSpace($ExecutablePath)) {
        return $false
    }
    foreach ($directory in $allowedDirectories) {
        if (Test-PathWithinDirectory -Path $ExecutablePath -Directory $directory) { return $true }
    }
    return $false
}

function Request-GracefulServerShutdown {
    param([System.Diagnostics.Process]$Process)

    if ($Process.ProcessName -ne 'ninfer-windows-serve') { return $false }
    $listeners = @(Get-NetTCPConnection -State Listen -OwningProcess $Process.Id -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $loopback = switch ($listener.LocalAddress) {
            '127.0.0.1' { '127.0.0.1' }
            '0.0.0.0' { '127.0.0.1' }
            '::1' { '[::1]' }
            '::' { '[::1]' }
            default { $null }
        }
        if (-not $loopback) { continue }
        $shutdownUri = 'http://{0}:{1}/ui/shutdown' -f $loopback, $listener.LocalPort
        $shutdownHeaders = @{'X-NInfer-Control' = '1'}
        if ($ApiKey) { $shutdownHeaders.Authorization = 'Bearer ' + $ApiKey }
        try {
            Invoke-RestMethod -Method Post -Uri $shutdownUri -Headers $shutdownHeaders -TimeoutSec 5 | Out-Null
        } catch {
            continue
        }
        Write-Host ('PID {0}: shutdown requested; waiting for pending cache saves.' -f $Process.Id)
        if ($Process.WaitForExit(30000)) { return $true }
        Write-Warning ('PID {0} did not finish shutdown within 30 seconds; forcing termination. The latest cache save may be incomplete.' -f $Process.Id)
        return $false
    }
    Write-Warning ('PID {0}: local shutdown was unavailable or rejected; forcing termination. The latest cache save may be incomplete.' -f $Process.Id)
    return $false
}

$filter = ($executableNames | ForEach-Object { "Name='$_'" }) -join ' OR '
try {
    $processRecords = @(Get-CimInstance -ClassName Win32_Process -Filter $filter -ErrorAction Stop)
} catch {
    throw ('Could not inspect process executable paths. No processes were stopped. {0}' -f $_.Exception.Message)
}

$targets = [System.Collections.Generic.List[object]]::new()
$unverifiedCount = 0
$otherLocationCount = 0
foreach ($record in $processRecords) {
    $path = [string]$record.ExecutablePath
    if ([string]::IsNullOrWhiteSpace($path)) {
        $unverifiedCount++
        continue
    }
    $path = ConvertTo-NormalizedPath $path
    if (Test-IsRepositoryNInferProcess -Name ([string]$record.Name) -ExecutablePath $path) {
        $targets.Add([pscustomobject]@{
            Id = [int]$record.ProcessId
            Name = [string]$record.Name
            Path = $path
        })
    } else {
        $otherLocationCount++
    }
}

if ($List) {
    Write-Host 'List mode: no processes will be stopped.'
    foreach ($target in $targets) {
        Write-Host ('PID {0}  {1}  {2}' -f $target.Id, $target.Name, $target.Path)
    }
    if ($targets.Count -eq 0) { Write-Host 'No matching NInfer process was found under this checkout.' }
    if ($otherLocationCount -gt 0) {
        Write-Host ('Ignored {0} same-named process(es) outside this checkout.' -f $otherLocationCount)
    }
    if ($unverifiedCount -gt 0) {
        Write-Host ('Skipped {0} NInfer-named process(es) because Windows did not provide an executable path.' -f $unverifiedCount)
    }
    return
}

if ($targets.Count -eq 0) {
    Write-Host 'No matching NInfer process was found under this checkout. No process was stopped.'
    if ($otherLocationCount -gt 0) {
        Write-Host ('Ignored {0} same-named process(es) outside this checkout.' -f $otherLocationCount)
    }
    if ($unverifiedCount -gt 0) {
        Write-Warning ('Skipped {0} NInfer-named process(es) because Windows did not provide an executable path.' -f $unverifiedCount)
    }
    return
}

Write-Host ('Stopping {0} NInfer process(es) from this checkout.' -f $targets.Count)
$stoppedCount = 0
$alreadyExitedCount = 0
$failedCount = 0
foreach ($target in $targets) {
    $process = $null
    try {
        $process = Get-Process -Id $target.Id -ErrorAction Stop
        $liveExecutableName = if ($process.Path) { [System.IO.Path]::GetFileName($process.Path) } else { '' }
        if (-not $process.Path -or
            (ConvertTo-NormalizedPath $process.Path) -ine $target.Path -or
            -not (Test-IsRepositoryNInferProcess -Name $liveExecutableName -ExecutablePath $process.Path)) {
            throw 'the PID no longer points to the same in-repository NInfer executable'
        }
        if (-not (Request-GracefulServerShutdown -Process $process)) {
            if (-not $process.HasExited) {
                Stop-Process -InputObject $process -Force -ErrorAction Stop
            }
            if (-not $process.WaitForExit(5000)) {
                throw 'termination was requested, but the process did not exit within five seconds'
            }
        }
        $stoppedCount++
        Write-Host ('Stopped PID {0} ({1}).' -f $target.Id, $target.Name)
    } catch {
        $liveProcess = Get-Process -Id $target.Id -ErrorAction SilentlyContinue
        if (-not $liveProcess) {
            $alreadyExitedCount++
            Write-Host ('PID {0} ({1}) had already exited.' -f $target.Id, $target.Name)
        } elseif ($liveProcess.Path -and (ConvertTo-NormalizedPath $liveProcess.Path) -ine $target.Path) {
            $alreadyExitedCount++
            Write-Host ('PID {0} no longer refers to the selected NInfer executable; it was left untouched.' -f $target.Id)
        } else {
            $failedCount++
            Write-Warning ('Could not stop PID {0} ({1}): {2}' -f $target.Id, $target.Name, $_.Exception.Message)
        }
    } finally {
        if ($process) { $process.Dispose() }
    }
}

Write-Host ('Finished: stopped {0}, already exited {1}, failed {2}.' -f $stoppedCount, $alreadyExitedCount, $failedCount)
if ($otherLocationCount -gt 0) {
    Write-Host ('Ignored {0} same-named process(es) outside this checkout.' -f $otherLocationCount)
}
if ($unverifiedCount -gt 0) {
    Write-Warning ('Skipped {0} NInfer-named process(es) because Windows did not provide an executable path.' -f $unverifiedCount)
}
if ($failedCount -gt 0) { exit 1 }
