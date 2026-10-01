[CmdletBinding()]
param(
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repoRoot 'models'
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot $OutputDirectory
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$manifestPath = Join-Path $PSScriptRoot 'model-v2.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

if ($manifest.repository -ne 'neroued/Qwen3.8-27B-nvfp4-NInfer' -or
    $manifest.revision -ne '52907138a5d23a8f7f868ba7b773e721fd275405' -or
    $manifest.upstream_file -ne 'qwen3_8_27b_nvfp4.ninfer' -or
    $manifest.local_file -ne 'qwen3_8_27b_nvfp4_v2.ninfer' -or
    [long]$manifest.size_bytes -ne 23719496192 -or
    $manifest.sha256 -ne '552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c') {
    throw 'The model manifest does not match the pinned V2 artifact identity.'
}

$curlCommand = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curlCommand) {
    throw 'curl.exe was not found. Install a Windows version that includes curl.exe, then retry.'
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$finalPath = Join-Path $OutputDirectory $manifest.local_file
$partialPath = $finalPath + '.partial'
$expectedHash = [string]$manifest.sha256
$expectedBytes = [long]$manifest.size_bytes

function Test-PinnedArtifact {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $file = Get-Item -LiteralPath $Path
    if ([long]$file.Length -ne $expectedBytes) { return $false }
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return $actualHash -eq $expectedHash
}

if (Test-Path -LiteralPath $finalPath -PathType Leaf) {
    if (Test-PinnedArtifact $finalPath) {
        Write-Host "Pinned V2 model is already present and verified: $finalPath"
        return
    }
    throw "The destination already exists but does not match the pinned V2 size and SHA256. It was left untouched: '$finalPath'."
}

if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
    $partialFile = Get-Item -LiteralPath $partialPath
    if ([long]$partialFile.Length -eq $expectedBytes) {
        if (Test-PinnedArtifact $partialPath) {
            Move-Item -LiteralPath $partialPath -Destination $finalPath
            Write-Host "Verified and finalized the completed V2 download: $finalPath"
            return
        }
        $quarantinePath = $partialPath + '.invalid.' + [guid]::NewGuid().ToString('N')
        Move-Item -LiteralPath $partialPath -Destination $quarantinePath
        Write-Warning "A full-size partial file failed SHA256 validation and was preserved as '$quarantinePath'."
    } elseif ([long]$partialFile.Length -gt $expectedBytes) {
        $quarantinePath = $partialPath + '.oversize.' + [guid]::NewGuid().ToString('N')
        Move-Item -LiteralPath $partialPath -Destination $quarantinePath
        Write-Warning "An oversized partial file was preserved as '$quarantinePath'."
    }
}

$downloadUrl = 'https://huggingface.co/' + $manifest.repository + '/resolve/' +
    $manifest.revision + '/' + $manifest.upstream_file + '?download=true'
Write-Host "Downloading the pinned V2 model to '$partialPath'. The 23.7 GB file may take a while."
& $curlCommand.Source --location --fail --retry 5 --retry-delay 2 --retry-connrefused --continue-at - --output $partialPath $downloadUrl
if ($LASTEXITCODE -ne 0) {
    throw "curl.exe failed with exit code $LASTEXITCODE. The partial file is kept so the next run can resume."
}

if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf)) {
    throw 'curl.exe returned success but did not create the partial model file.'
}
$downloadedBytes = [long](Get-Item -LiteralPath $partialPath).Length
if ($downloadedBytes -ne $expectedBytes) {
    throw "Downloaded file size is $downloadedBytes bytes; expected $expectedBytes. The partial file is kept."
}
$downloadedHash = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($downloadedHash -ne $expectedHash) {
    throw "Downloaded model SHA256 does not match the pinned V2 artifact. The partial file is kept: '$partialPath'."
}

Move-Item -LiteralPath $partialPath -Destination $finalPath
Write-Host "Pinned V2 model downloaded and verified: $finalPath"
