<#
.SYNOPSIS
Builds hush.exe and hush-stt-worker.exe in release, checks that they load nothing but
system DLLs, the VC++ runtime and the Vulkan loader, and zips them with the licences and
a README into target\release\hush-<version>-windows-x64.zip.

.PARAMETER NoBuild
Package what is already in target\release.

.NOTES
Runs in Windows PowerShell 5.1 and PowerShell 7. Keep this file ASCII: 5.1 reads a .ps1
without a BOM in the ANSI code page.
#>
param([switch]$NoBuild)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$release = Join-Path $root 'target\release'
$exes = @('hush.exe', 'hush-stt-worker.exe')

function Write-Utf8([string]$path, [string]$text) {
    [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))
}

function Find-Dumpbin {
    $onPath = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe' | Select-Object -First 1
        if ($found) { return $found }
    }
    throw 'dumpbin.exe not found; install the MSVC build tools'
}

# Load-time and delay-load imports alike.
function Get-Dependents([string]$dumpbin, [string]$exe) {
    $lines = & $dumpbin /nologo /dependents $exe
    if ($LASTEXITCODE -ne 0) { throw "dumpbin failed on $exe" }
    $lines | ForEach-Object { $_.Trim() } |
        Where-Object { $_ -match '^[A-Za-z0-9_.\-]+\.dll$' } |
        Sort-Object -Unique
}

# $null means the DLL would have to ship with hush, which the bundle does not allow.
function Get-DllKind([string]$name) {
    $n = $name.ToLowerInvariant()
    if ($n -match '^(vcruntime140(_1)?|msvcp140(_1|_2|_atomic_wait|_codecvt_ids)?|concrt140|vcomp140)\.dll$') {
        return 'VC++ runtime'
    }
    if ($n -eq 'vulkan-1.dll') { return 'Vulkan loader' }
    if ($n -match '^(api|ext)-ms-win-') { return 'system (API set)' }
    $path = Join-Path $env:SystemRoot "System32\$n"
    if (Test-Path $path) {
        $sig = Get-AuthenticodeSignature $path
        if ($sig.Status -eq 'Valid' -and $sig.SignerCertificate.Subject -match 'O=Microsoft Corporation') {
            return 'system'
        }
    }
    return $null
}

function Get-WorkspaceVersion {
    $m = Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' |
        Select-Object -First 1
    if (-not $m) { throw 'no version in the workspace Cargo.toml' }
    $m.Matches[0].Groups[1].Value
}

function Get-ModelNotices {
    $ids = Select-String -Path (Join-Path $root 'crates\stt\src\models.rs') `
        -Pattern 'pub const (DEFAULT_MODEL_ID|VAD_MODEL_ID|DEFAULT_LLM_ID): &str = "([^"]+)"' |
        ForEach-Object { $_.Matches[0].Groups[2].Value }
    if (@($ids).Count -ne 3) { throw "expected the speech, VAD and LLM model ids, found: $ids" }
    $manifest = Get-Content -Raw -Encoding UTF8 (Join-Path $root 'crates\stt\src\models.json') | ConvertFrom-Json
    foreach ($id in $ids) {
        $m = $manifest.models | Where-Object { $_.id -eq $id }
        if (-not $m) { throw "model $id is not in the manifest" }
        "- **$($m.id)** ($($m.license)): $($m.attribution)"
    }
}

# Normal edges only, with features resolved for these two packages alone, so a crate that
# only a bench tool or a test pulls in is not listed.
function Get-CrateLicenses {
    $lines = & cargo tree -p hush -p hush-stt-worker -e normal --prefix none `
        --target x86_64-pc-windows-msvc -f '{p}|{l}'
    if ($LASTEXITCODE -ne 0) { throw 'cargo tree failed' }
    $lines | Where-Object { $_ -match '\|' -and $_ -notmatch '\(\S:\\' } |
        ForEach-Object { ($_ -replace ' \(\*\)$', '').Trim() } |
        Sort-Object -Unique |
        ForEach-Object {
            $p, $l = $_ -split '\|', 2
            if (-not $l) { $l = 'no licence field' }
            [pscustomobject]@{ Crate = $p.Trim(); License = $l.Trim() }
        }
}

Push-Location $root
try {
    if (-not $NoBuild) {
        & cargo build --release -p hush -p hush-stt-worker
        if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
    }
    foreach ($e in $exes) {
        $p = Join-Path $release $e
        if (-not (Test-Path $p)) { throw "missing $p" }
    }

    Write-Host 'dependency check'
    $dumpbin = Find-Dumpbin
    $bad = @()
    foreach ($e in $exes) {
        Write-Host "  $e"
        foreach ($dll in Get-Dependents $dumpbin (Join-Path $release $e)) {
            $kind = Get-DllKind $dll
            if ($kind) {
                Write-Host ('    {0,-44} {1}' -f $dll, $kind)
            } else {
                Write-Host ('    {0,-44} NOT ALLOWED' -f $dll)
                $bad += "$e -> $dll"
            }
        }
    }
    if ($bad.Count -gt 0) { throw "unexpected DLL dependencies: $($bad -join ', ')" }

    $version = Get-WorkspaceVersion
    $name = "hush-$version-windows-x64"
    $stage = Join-Path $release $name
    $zip = Join-Path $release "$name.zip"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory $stage | Out-Null
    foreach ($e in $exes) { Copy-Item (Join-Path $release $e) $stage }
    Copy-Item (Join-Path $root 'LICENSE') $stage
    Copy-Item (Join-Path $PSScriptRoot 'README.txt') $stage

    $crates = @(Get-CrateLicenses)
    $groups = $crates | Group-Object License | Sort-Object @{ e = 'Count'; Descending = $true }, Name
    $notice = New-Object Text.StringBuilder
    [void]$notice.AppendLine("# Third-party notices for hush $version")
    [void]$notice.AppendLine()
    [void]$notice.AppendLine('hush itself is MIT licensed; see LICENSE.')
    [void]$notice.AppendLine()
    [void]$notice.AppendLine('## Models')
    [void]$notice.AppendLine()
    [void]$notice.AppendLine('Not in this archive: hush downloads them on first run from a manifest compiled into it.')
    [void]$notice.AppendLine('Their licences apply to the downloaded files.')
    [void]$notice.AppendLine()
    foreach ($line in Get-ModelNotices) { [void]$notice.AppendLine($line) }
    [void]$notice.AppendLine()
    [void]$notice.AppendLine('## Native libraries compiled into the executables')
    [void]$notice.AppendLine()
    [void]$notice.AppendLine('- llama.cpp and its ggml, in hush.exe: MIT, Copyright (c) 2023-2026 The ggml authors.')
    [void]$notice.AppendLine('- transcribe.cpp, in hush-stt-worker.exe: MIT, Copyright (c) 2026 The transcribe.cpp authors;')
    [void]$notice.AppendLine('  its ggml: MIT, Copyright (c) 2023-2026 The ggml authors.')
    [void]$notice.AppendLine()
    [void]$notice.AppendLine('## Rust crates')
    [void]$notice.AppendLine()
    [void]$notice.AppendLine("The $($crates.Count) crates linked into hush.exe and hush-stt-worker.exe, by licence")
    [void]$notice.AppendLine('expression as each crate declares it (generated with `cargo tree`).')
    foreach ($g in $groups) {
        [void]$notice.AppendLine()
        [void]$notice.AppendLine("### $($g.Name) ($($g.Count))")
        [void]$notice.AppendLine()
        [void]$notice.AppendLine((($g.Group | ForEach-Object { $_.Crate }) -join ', '))
    }
    Write-Utf8 (Join-Path $stage 'THIRD-PARTY.md') $notice.ToString()

    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
    Remove-Item -Recurse -Force $stage

    Write-Host ''
    Write-Host "wrote $zip"
    Write-Host ('  {0:N1} MB' -f ((Get-Item $zip).Length / 1MB))
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($zip)
    try {
        foreach ($entry in $archive.Entries) {
            Write-Host ('  {0,-24} {1,12:N0} bytes' -f $entry.FullName, $entry.Length)
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    Pop-Location
}
