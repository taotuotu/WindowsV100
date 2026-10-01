[CmdletBinding()]
param(
    [ValidateSet('configure', 'core', 'ops', 'cli', 'server', 'all')]
    [string]$Step = 'all',
    [string]$BuildDirectory = '',
    [string]$CMakePath = '',
    [string]$CudaRoot = '',
    [ValidateRange(1, 32)]
    [int]$Parallel = 4
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $repoRoot 'build-win-v100'
} elseif (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)

if (-not $CMakePath) {
    $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmakeCommand) {
        $CMakePath = $cmakeCommand.Source
    } else {
        $candidates = @(
            (Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe')
        )
        $pythonRoots = Join-Path $env:LOCALAPPDATA 'Python'
        if (Test-Path -LiteralPath $pythonRoots -PathType Container) {
            $candidates += Get-ChildItem -LiteralPath $pythonRoots -Directory -Filter 'pythoncore-*' |
                ForEach-Object { Join-Path $_.FullName 'Lib\site-packages\cmake\data\bin\cmake.exe' }
        }
        $CMakePath = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
            Select-Object -First 1
    }
}
if (-not $CMakePath -or -not (Test-Path -LiteralPath $CMakePath -PathType Leaf)) {
    throw 'CMake 3.28+ was not found. Pass -CMakePath <cmake.exe>.'
}
$CMakePath = (Resolve-Path -LiteralPath $CMakePath).Path

function Get-CudaRootCandidates {
    $roots = [System.Collections.Generic.List[string]]::new()
    if ($CudaRoot) { $roots.Add($CudaRoot) }
    if ($env:CUDA_PATH) { $roots.Add($env:CUDA_PATH) }

    Get-ChildItem Env: | Where-Object { $_.Name -match '^CUDA_PATH_V12_' } |
        Sort-Object Name -Descending | ForEach-Object { $roots.Add($_.Value) }

    if ($env:ProgramFiles) {
        $cudaInstallRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
        if (Test-Path -LiteralPath $cudaInstallRoot -PathType Container) {
            Get-ChildItem -LiteralPath $cudaInstallRoot -Directory |
                Where-Object { $_.Name -match '^v12\.\d+$' } |
                Sort-Object { [version]($_.Name.Substring(1)) } -Descending |
                ForEach-Object { $roots.Add($_.FullName) }
        }
    }

    return @($roots | Where-Object { $_ } | Select-Object -Unique)
}

$cudaRootResolved = $null
$nvcc = $null
$toolkitVersion = $null
foreach ($candidateRoot in (Get-CudaRootCandidates)) {
    $candidateNvcc = Join-Path $candidateRoot 'bin\nvcc.exe'
    if (-not (Test-Path -LiteralPath $candidateNvcc -PathType Leaf)) { continue }

    $nvccVersionOutput = (& $candidateNvcc --version 2>&1 | Out-String)
    $versionMatch = [regex]::Match($nvccVersionOutput, 'release\s+(\d+\.\d+)')
    if ($LASTEXITCODE -ne 0 -or -not $versionMatch.Success) { continue }
    $versionText = $versionMatch.Groups[1].Value
    $candidateVersion = [version]$versionText
    if ($candidateVersion -lt [version]'12.8' -or $candidateVersion -ge [version]'13.0') { continue }

    $cudaRootResolved = (Resolve-Path -LiteralPath $candidateRoot).Path
    $nvcc = (Resolve-Path -LiteralPath $candidateNvcc).Path
    $toolkitVersion = $versionText
    break
}
if (-not $nvcc) {
    throw 'A CUDA Toolkit version from 12.8 up to (but not including) 13.0 was not found. Set CUDA_PATH or pass -CudaRoot <CUDA 12.x directory>.'
}

New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null
$log = Join-Path $BuildDirectory 'build-windows-v100.log'

function Invoke-CMakeLogged([string[]]$Arguments) {
    & $CMakePath @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "CMake failed with exit code $LASTEXITCODE; see '$log'."
    }
}

if ($Step -in @('configure', 'server', 'all')) {
    Invoke-CMakeLogged @(
        '-S', $repoRoot,
        '-B', $BuildDirectory,
        '-G', 'Visual Studio 17 2022',
        '-A', 'x64',
        '-T', "cuda=$toolkitVersion",
        "-DCMAKE_CUDA_COMPILER=$nvcc",
        '-DCMAKE_CUDA_ARCHITECTURES=70',
        '-DNINFER_BUILD_APPS=OFF',
        '-DNINFER_BUILD_WINDOWS_TEXT=ON',
        '-DNINFER_BUILD_WINDOWS_SERVE=ON',
        '-DBUILD_TESTING=OFF',
        '-DNINFER_BUILD_BENCHMARKS=OFF'
    )
}

$targets = switch ($Step) {
    'core' { @('ninfer_core') }
    'ops'  { @('ninfer_ops') }
    'cli'  { @('ninfer-windows-text') }
    'server' { @('ninfer-windows-serve') }
    'all'  { @('ninfer_core', 'ninfer_ops', 'ninfer-windows-text', 'ninfer-windows-serve') }
    default { @() }
}
foreach ($target in $targets) {
    Invoke-CMakeLogged @(
        '--build', $BuildDirectory,
        '--target', $target,
        '--config', 'Release',
        '--parallel', "$Parallel"
    )
}
