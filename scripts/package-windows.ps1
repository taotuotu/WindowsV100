[CmdletBinding()]
param(
    [ValidateSet('binary', 'source', 'both')]
    [string]$Package = 'both',
    [string]$OutputDirectory = '',
    [string]$BuildDirectory = '',
    [string]$CudaRoot = '',
    [switch]$IncludeTextCli,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$createdArchives = [System.Collections.Generic.List[string]]::new()
$gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
if (-not $gitCommand) { $gitCommand = Get-Command git -ErrorAction SilentlyContinue }
$gitExecutable = if ($gitCommand) { $gitCommand.Source } else { $null }
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repoRoot 'dist'
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot $OutputDirectory
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $repoRoot 'build-win-v100'
} elseif (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)

$excludedDirectoryNames = @(
    '.git', '.local', '.codex', 'build', 'dist', 'logs', 'results', 'models', 'profiles'
)
$excludedExtensions = @('.ninfer', '.gguf', '.safetensors', '.log', '.nsys-rep', '.ncu-rep', '.snap', '.head', '.tmp')
$rootFiles = @(
    '.clang-format', '.clangd', '.dockerignore', '.gitattributes', '.gitignore',
    'AGENTS.md', 'CMakeLists.txt', 'CONTRIBUTING.md', 'Dockerfile', 'LICENSE', 'NOTICE',
    'README.md', 'THIRD_PARTY_NOTICES.md',
    'start-ninfer.bat', 'stop-ninfer.bat', 'open-chat.bat'
)
$sourceDirectories = @(
    '.github', 'apps', 'bench', 'cmake', 'docs', 'eval', 'examples',
    'include', 'licenses', 'model-cards', 'scripts', 'src', 'tests',
    'third_party', 'tools'
)

function Invoke-GitReadOnly {
    param([string[]]$Arguments)
    if (-not $script:gitExecutable) {
        throw 'git was not found; release metadata and the source archive require git.'
    }
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $standardOutput = @(& $script:gitExecutable @Arguments 2>$null)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        $commandText = 'git ' + ($Arguments -join ' ')
        throw ('Git command failed with exit code {0}: {1}.' -f $exitCode, $commandText)
    }
    return $standardOutput
}

function Test-PackagePathAllowed {
    param([string]$RelativePath)
    $normalized = $RelativePath.Replace('\', '/').TrimStart('/')
    if (-not $normalized) { return $false }
    $segments = @($normalized.Split('/'))
    foreach ($segment in $segments) {
        if ($excludedDirectoryNames -contains $segment.ToLowerInvariant()) { return $false }
    }
    $leaf = $segments[-1]
    if ($leaf.StartsWith('ninfer-session-cache-', [StringComparison]::OrdinalIgnoreCase)) { return $false }
    if ($leaf.EndsWith('.requests.jsonl', [StringComparison]::OrdinalIgnoreCase)) { return $false }
    if ($excludedExtensions -contains [System.IO.Path]::GetExtension($leaf).ToLowerInvariant()) { return $false }
    if ($segments.Count -eq 1) { return $rootFiles -contains $leaf }
    return $sourceDirectories -contains $segments[0]
}

function Add-PackageFile {
    param(
        [System.Collections.Generic.List[object]]$Files,
        [string]$SourcePath,
        [string]$ArchivePath
    )
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        throw "Required package file does not exist: '$SourcePath'."
    }
    $normalizedArchivePath = $ArchivePath.Replace('\', '/').TrimStart('/')
    if (-not $normalizedArchivePath -or $normalizedArchivePath.StartsWith('../')) {
        throw "Invalid archive path '$ArchivePath'."
    }
    $Files.Add([pscustomobject]@{
        SourcePath = (Resolve-Path -LiteralPath $SourcePath).Path
        ArchivePath = $normalizedArchivePath
    }) | Out-Null
}

function Add-DirectoryFiles {
    param(
        [System.Collections.Generic.List[object]]$Files,
        [string]$SourceDirectory,
        [string]$ArchivePrefix,
        [switch]$Required
    )
    if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
        if ($Required) { throw "Required package directory does not exist: '$SourceDirectory'." }
        return
    }
    $resolvedDirectory = (Resolve-Path -LiteralPath $SourceDirectory).Path.TrimEnd('\', '/')
    foreach ($file in Get-ChildItem -LiteralPath $resolvedDirectory -File -Recurse -Force) {
        $relative = $file.FullName.Substring($resolvedDirectory.Length).TrimStart('\', '/')
        $archivePath = if ($ArchivePrefix) { Join-Path $ArchivePrefix $relative } else { $relative }
        if (Test-PackagePathAllowed $archivePath) {
            Add-PackageFile $Files $file.FullName $archivePath
        }
    }
}

function New-DeterministicZip {
    param(
        [System.Collections.Generic.List[object]]$Files,
        [string]$DestinationPath
    )
    Add-Type -AssemblyName System.IO.Compression
    $stream = [System.IO.File]::Open($DestinationPath, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $archive = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Create, $false)
    try {
        foreach ($file in @($Files | Sort-Object ArchivePath -Unique)) {
            $entry = $archive.CreateEntry($file.ArchivePath, [System.IO.Compression.CompressionLevel]::Optimal)
            $entry.LastWriteTime = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
            $inputStream = [System.IO.File]::OpenRead($file.SourcePath)
            $entryStream = $entry.Open()
            try {
                $inputStream.CopyTo($entryStream)
            } finally {
                $entryStream.Dispose()
                $inputStream.Dispose()
            }
        }
    } finally {
        $archive.Dispose()
        $stream.Dispose()
    }
}

function Publish-Package {
    param(
        [System.Collections.Generic.List[object]]$Files,
        [string]$ArchiveName
    )
    if ($Files.Count -eq 0) { throw "No files were selected for '$ArchiveName'." }
    $destinationPath = Join-Path $OutputDirectory $ArchiveName
    if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
        if (-not $Force) { throw "Package already exists: '$destinationPath'. Pass -Force to replace this generated archive." }
        $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        $resolvedDestination = [System.IO.Path]::GetFullPath($destinationPath)
        if (-not $resolvedDestination.StartsWith($resolvedOutput, [StringComparison]::OrdinalIgnoreCase) -or
            [System.IO.Path]::GetFileName($resolvedDestination) -ne $ArchiveName) {
            throw "Refusing to replace a package outside the selected output directory: '$resolvedDestination'."
        }
        Remove-Item -LiteralPath $resolvedDestination -Force
    }
    New-DeterministicZip $Files $destinationPath
    $script:createdArchives.Add($destinationPath) | Out-Null
    Write-Host "Created $destinationPath ($($Files.Count) allowlisted files)."
}

function Write-ReleaseManifest {
    param(
        [string]$ServerExecutable,
        [string]$CudaRuntime
    )
    $sourceCommit = $null
    $worktreeDirty = $null
    $sourceChanges = [System.Collections.Generic.List[object]]::new()
    if ($gitExecutable) {
        $sourceCommit = Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', 'HEAD') |
            Select-Object -First 1
        $changedPaths = @(Invoke-GitReadOnly @('-C', $repoRoot, 'diff', '--name-only', 'HEAD', '--'))
        $untrackedPaths = @(Invoke-GitReadOnly @('-C', $repoRoot, 'ls-files', '--others', '--exclude-standard'))
        $allChangedPaths = @($changedPaths) + @($untrackedPaths)
        foreach ($relativePath in ($allChangedPaths | Sort-Object -Unique)) {
            if (-not (Test-PackagePathAllowed ([string]$relativePath))) { continue }
            $sourcePath = Join-Path $repoRoot ([string]$relativePath)
            $normalizedPath = ([string]$relativePath).Replace('\', '/')
            $fileHash = $null
            if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
                $fileHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            $sourceChanges.Add([pscustomobject]@{
                path = $normalizedPath
                sha256 = $fileHash
            }) | Out-Null
        }
        $worktreeDirty = $sourceChanges.Count -gt 0
    }

    $cachePath = Join-Path $BuildDirectory 'CMakeCache.txt'
    $cache = @{}
    if (Test-Path -LiteralPath $cachePath -PathType Leaf) {
        foreach ($line in Get-Content -LiteralPath $cachePath) {
            if ($line -match '^([^:#]+):[^=]*=(.*)$') { $cache[$Matches[1]] = $Matches[2] }
        }
    }
    $cudaCompilerVersion = $cache['CMAKE_CUDA_COMPILER_VERSION']
    $cxxCompilerVersion = $cache['CMAKE_CXX_COMPILER_VERSION']
    $cxxCompilerId = $cache['CMAKE_CXX_COMPILER_ID']
    $platformToolsetVersion = $cache['CMAKE_VS_PLATFORM_TOOLSET_VERSION']
    $cmakeFilesDirectory = Join-Path $BuildDirectory 'CMakeFiles'
    if (Test-Path -LiteralPath $cmakeFilesDirectory -PathType Container) {
        foreach ($versionDirectory in (Get-ChildItem -LiteralPath $cmakeFilesDirectory -Directory |
                Sort-Object Name -Descending)) {
            $cudaCompilerFile = Join-Path $versionDirectory.FullName 'CMakeCUDACompiler.cmake'
            if (-not $cudaCompilerVersion -and (Test-Path -LiteralPath $cudaCompilerFile -PathType Leaf)) {
                foreach ($line in Get-Content -LiteralPath $cudaCompilerFile) {
                    if ($line -match '^set\(CMAKE_CUDA_COMPILER_VERSION "([^"]+)"\)') {
                        $cudaCompilerVersion = $Matches[1]
                        break
                    }
                }
            }
            $cxxCompilerFile = Join-Path $versionDirectory.FullName 'CMakeCXXCompiler.cmake'
            if ((-not $cxxCompilerVersion -or -not $cxxCompilerId) -and
                (Test-Path -LiteralPath $cxxCompilerFile -PathType Leaf)) {
                foreach ($line in Get-Content -LiteralPath $cxxCompilerFile) {
                    if (-not $cxxCompilerVersion -and $line -match '^set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)') {
                        $cxxCompilerVersion = $Matches[1]
                    }
                    if (-not $cxxCompilerId -and $line -match '^set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)') {
                        $cxxCompilerId = $Matches[1]
                    }
                }
            }
            if ($cudaCompilerVersion -and $cxxCompilerVersion -and $cxxCompilerId) { break }
        }
    }

    $modelManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'model-v2.json') -Raw |
        ConvertFrom-Json
    $serverHash = $null
    $runtimeHash = $null
    $runtimeFileVersion = $null
    if ($ServerExecutable -and (Test-Path -LiteralPath $ServerExecutable -PathType Leaf)) {
        $serverHash = (Get-FileHash -LiteralPath $ServerExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($CudaRuntime -and (Test-Path -LiteralPath $CudaRuntime -PathType Leaf)) {
        $runtimeHash = (Get-FileHash -LiteralPath $CudaRuntime -Algorithm SHA256).Hash.ToLowerInvariant()
        $runtimeFileVersion = (Get-Item -LiteralPath $CudaRuntime).VersionInfo.FileVersion
    }

    $manifest = [ordered]@{
        schema_version = 1
        project = 'NInfer Windows V100'
        base_commit = $sourceCommit
        source_worktree_dirty = $worktreeDirty
        source_changed_files = @($sourceChanges | Sort-Object path)
        build = [ordered]@{
            platform = 'Windows x64'
            configuration = 'Release'
            target = 'ninfer-windows-serve'
            generator = $cache['CMAKE_GENERATOR']
            cuda_toolkit_version = $cudaCompilerVersion
            cuda_architectures = $cache['CMAKE_CUDA_ARCHITECTURES']
            cxx_compiler_id = $cxxCompilerId
            cxx_compiler_version = $cxxCompilerVersion
            vs_platform_toolset_version = $platformToolsetVersion
            runtime_library = 'cudart64_12.dll'
            bundled_cuda_runtime = '12.9.x'
        }
        binary = [ordered]@{
            server_sha256 = $serverHash
            cudart_sha256 = $runtimeHash
            cudart_bundled = $null -ne $runtimeHash
            cudart_file_version = $runtimeFileVersion
            cudart_product_version = if ($CudaRuntime -and (Test-Path -LiteralPath $CudaRuntime -PathType Leaf)) {
                (Get-Item -LiteralPath $CudaRuntime).VersionInfo.ProductVersion
            } else { $null }
            driver_dll_bundled = $false
            model_bundled = $false
        }
        model = [ordered]@{
            repository = $modelManifest.repository
            revision = $modelManifest.revision
            file = $modelManifest.upstream_file
            bytes = [long]$modelManifest.size_bytes
            sha256 = $modelManifest.sha256
            bundled = $false
        }
        public_defaults = [ordered]@{
            device = 'auto (first SM70 device with at least 30 GiB)'
            context_tokens = 143600
            kv_dtype = 'bf16'
            mtp_draft_tokens = 6
            prefill_chunk_tokens = 2048
            max_concurrency = 1
            disk_cache_enabled = $true
            disk_cache_directory = '.local/context-cache'
            disk_cache_mib = 32768
            disk_cache_sessions = 8
        }
    }
    return $manifest
}

function Save-ReleaseManifest {
    param([object]$Manifest)
    $manifestPath = Join-Path $OutputDirectory 'release-manifest.json'
    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        if (-not $Force) {
            throw "Manifest already exists: '$manifestPath'. Pass -Force to replace generated release metadata."
        }
        $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        $resolvedManifest = [System.IO.Path]::GetFullPath($manifestPath)
        if (-not $resolvedManifest.StartsWith($resolvedOutput, [StringComparison]::OrdinalIgnoreCase) -or
            [System.IO.Path]::GetFileName($resolvedManifest) -ne 'release-manifest.json') {
            throw "Refusing to replace release metadata outside the selected output directory: '$resolvedManifest'."
        }
        Remove-Item -LiteralPath $resolvedManifest -Force
    }
    $json = $Manifest | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText($manifestPath, $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false))
    return $manifestPath
}

function Write-ChecksumManifest {
    $checksumPath = Join-Path $OutputDirectory 'SHA256SUMS.txt'
    if (Test-Path -LiteralPath $checksumPath -PathType Leaf) {
        if (-not $Force) {
            throw "Checksum manifest already exists: '$checksumPath'. Pass -Force to replace generated release metadata."
        }
        $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        $resolvedChecksum = [System.IO.Path]::GetFullPath($checksumPath)
        if (-not $resolvedChecksum.StartsWith($resolvedOutput, [StringComparison]::OrdinalIgnoreCase) -or
            [System.IO.Path]::GetFileName($resolvedChecksum) -ne 'SHA256SUMS.txt') {
            throw "Refusing to replace checksum metadata outside the selected output directory: '$resolvedChecksum'."
        }
        Remove-Item -LiteralPath $resolvedChecksum -Force
    }
    $lines = foreach ($archivePath in $createdArchives) {
        $hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        '{0}  {1}' -f $hash, [System.IO.Path]::GetFileName($archivePath)
    }
    [System.IO.File]::WriteAllLines($checksumPath, [string[]]$lines,
        [System.Text.UTF8Encoding]::new($false))
    Write-Host "Wrote package checksums to $checksumPath."
}

function Get-CudaRuntimeDll {
    $directories = [System.Collections.Generic.List[string]]::new()
    if ($CudaRoot) {
        if (-not [System.IO.Path]::IsPathRooted($CudaRoot)) {
            $CudaRoot = Join-Path (Get-Location).Path $CudaRoot
        }
        $explicitRoot = [System.IO.Path]::GetFullPath($CudaRoot)
        $explicitBin = if ((Split-Path -Leaf $explicitRoot) -ieq 'bin') {
            $explicitRoot
        } else {
            Join-Path $explicitRoot 'bin'
        }
        $explicitDll = Join-Path $explicitBin 'cudart64_12.dll'
        if (-not (Test-Path -LiteralPath $explicitDll -PathType Leaf)) {
            throw "The explicit CUDA root does not contain cudart64_12.dll: '$explicitBin'."
        }
        $item = Get-Item -LiteralPath $explicitDll
        $versionText = [string]$item.VersionInfo.FileVersion
        $productText = [string]$item.VersionInfo.ProductVersion
        if ($versionText -notmatch '(?:^|[,.\s])1209\d+$' -and
            $productText -notmatch '(?:^|[,.\s])1209\d+$') {
            throw "The explicit CUDA runtime is not a CUDA 12.9.x DLL (FileVersion='$versionText', ProductVersion='$productText')."
        }
        return [pscustomobject]@{
            Path = (Resolve-Path -LiteralPath $explicitDll).Path
            FileVersion = $versionText
            ProductVersion = $productText
        }
    }

    $releaseBin = Join-Path $repoRoot 'bin'
    if (Test-Path -LiteralPath (Join-Path $releaseBin 'cudart64_12.dll') -PathType Leaf) {
        $directories.Add($releaseBin)
    }
    $buildCache = Join-Path $BuildDirectory 'CMakeCache.txt'
    if (Test-Path -LiteralPath $buildCache -PathType Leaf) {
        foreach ($line in Get-Content -LiteralPath $buildCache) {
            if ($line -match '^CMAKE_CUDA_COMPILER:[^=]*=(.*)$' -and $Matches[1]) {
                $compilerPath = $Matches[1]
                if (Test-Path -LiteralPath $compilerPath -PathType Leaf) {
                    $directories.Add((Split-Path -Parent $compilerPath))
                }
            } elseif ($line -match '^CUDAToolkit_ROOT:[^=]*=(.*)$' -and $Matches[1]) {
                $root = $Matches[1]
                if ((Split-Path -Leaf $root) -ieq 'bin') { $directories.Add($root) }
                else { $directories.Add((Join-Path $root 'bin')) }
            }
        }
    }
    if ($env:CUDA_PATH_V12_9) { $directories.Add((Join-Path $env:CUDA_PATH_V12_9 'bin')) }
    if ($env:CUDA_PATH) { $directories.Add((Join-Path $env:CUDA_PATH 'bin')) }
    Get-ChildItem Env: | Where-Object { $_.Name -match '^CUDA_PATH_V12_' } |
        ForEach-Object { $directories.Add((Join-Path $_.Value 'bin')) }
    if ($env:ProgramFiles) {
        $cudaInstallRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
        if (Test-Path -LiteralPath $cudaInstallRoot -PathType Container) {
            Get-ChildItem -LiteralPath $cudaInstallRoot -Directory |
                Where-Object { $_.Name -eq 'v12.9' } |
                ForEach-Object { $directories.Add((Join-Path $_.FullName 'bin')) }
        }
    }
    $directories.AddRange([string[]]($env:PATH -split ';' | Where-Object { $_ }))
    foreach ($directory in $directories) {
        $candidate = Join-Path $directory 'cudart64_12.dll'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $item = Get-Item -LiteralPath $candidate
            $versionText = [string]$item.VersionInfo.FileVersion
            $productText = [string]$item.VersionInfo.ProductVersion
            if ($versionText -match '(?:^|[,.\s])1209\d+$' -or
                $productText -match '(?:^|[,.\s])1209\d+$') {
                return [pscustomobject]@{
                    Path = (Resolve-Path -LiteralPath $candidate).Path
                    FileVersion = $versionText
                    ProductVersion = $productText
                }
            }
        }
    }
    throw 'A CUDA 12.9.x cudart64_12.dll was not found. Use -CudaRoot, CUDA_PATH_V12_9, or a CUDA 12.9 toolkit install.'
}

if ($Package -in @('binary', 'both')) {
    $serverExe = Join-Path $BuildDirectory 'apps\windows-serve\Release\ninfer-windows-serve.exe'
    if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
        $serverExe = Join-Path $repoRoot 'bin\ninfer-windows-serve.exe'
    }
    if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
        throw "The Windows server executable was not found in '$BuildDirectory' or '$repoRoot\bin'. Build it with scripts\build-windows-v100.ps1 -Step server."
    }
    $runtimeDll = Get-CudaRuntimeDll

    $manifestPath = Save-ReleaseManifest (Write-ReleaseManifest $serverExe $runtimeDll.Path)

    foreach ($requiredFile in @('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'README.md')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $requiredFile) -PathType Leaf)) {
            throw "Required binary release notice is missing: '$requiredFile'."
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'licenses') -PathType Container)) {
        throw 'Required third-party licenses directory is missing.'
    }

    $binaryFiles = [System.Collections.Generic.List[object]]::new()
    Add-PackageFile $binaryFiles $serverExe 'bin\ninfer-windows-serve.exe'
    Add-PackageFile $binaryFiles $runtimeDll.Path 'bin\cudart64_12.dll'
    if ($IncludeTextCli) {
        $textExe = Join-Path $BuildDirectory 'apps\windows-text\Release\ninfer-windows-text.exe'
        if (-not (Test-Path -LiteralPath $textExe -PathType Leaf)) {
            throw "The optional text CLI was requested but was not found: '$textExe'."
        }
        Add-PackageFile $binaryFiles $textExe 'bin\ninfer-windows-text.exe'
    }
    foreach ($rootFile in @(
        'README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md',
        'start-ninfer.bat', 'stop-ninfer.bat', 'open-chat.bat',
        'scripts\run-ninfer-server.ps1', 'scripts\stop-ninfer.ps1',
        'scripts\download-model.ps1', 'scripts\model-v2.json'
    )) {
        Add-PackageFile $binaryFiles (Join-Path $repoRoot $rootFile) $rootFile
    }
    Add-PackageFile $binaryFiles $manifestPath 'release-manifest.json'
    Add-DirectoryFiles $binaryFiles (Join-Path $repoRoot 'docs') 'docs' -Required
    Add-DirectoryFiles $binaryFiles (Join-Path $repoRoot 'licenses') 'licenses' -Required

    Publish-Package $binaryFiles 'ninfer-windows-v100-binary.zip'
}

if ($Package -in @('source', 'both')) {
    if (-not $gitExecutable) { throw 'git was not found; the source archive uses git ls-files for its allowlist.' }
    $sourcePaths = @(Invoke-GitReadOnly @('-C', $repoRoot, 'ls-files', '--cached', '--others', '--exclude-standard'))
    $sourceFiles = [System.Collections.Generic.List[object]]::new()
    if (-not $manifestPath) { $manifestPath = Save-ReleaseManifest (Write-ReleaseManifest $null $null) }
    foreach ($relativePath in $sourcePaths) {
        if (-not (Test-PackagePathAllowed ([string]$relativePath))) { continue }
        $sourcePath = Join-Path $repoRoot ([string]$relativePath)
        if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
            Add-PackageFile $sourceFiles $sourcePath ([string]$relativePath)
        }
    }

    foreach ($requiredFile in @(
        'CMakeLists.txt', 'README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md',
        'start-ninfer.bat', 'stop-ninfer.bat', 'open-chat.bat'
    )) {
        if (-not ($sourceFiles | Where-Object { $_.ArchivePath -eq $requiredFile })) {
            throw "Required source release file was not selected: '$requiredFile'."
        }
    }
    Add-PackageFile $sourceFiles $manifestPath 'release-manifest.json'
    Publish-Package $sourceFiles 'ninfer-windows-v100-source.zip'
}

if ($createdArchives.Count -gt 0) { Write-ChecksumManifest }
