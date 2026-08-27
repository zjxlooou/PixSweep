# PixSweep 构建脚本：前端 build + 后端 cargo build（无 MSVC，用 .tools zig + xwin 工具链）。
# 产物：src-tauri/target/release/pixsweep.exe（-Debug 时为 target/debug/）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts/build.ps1          # release
#   powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Debug   # debug
#
# 与 build_release.ps1 的区别：本脚本只构建可执行文件，不打包 zip、不复制模型/DLL。

param(
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Set-Location ".."
$root = (Get-Location).Path

$profile = if ($Debug) { "debug" } else { "release" }
Write-Host "=== PixSweep 构建（$profile）===" -ForegroundColor Cyan
Write-Host "项目根: $root"

# 1. 构建前端（Tauri 编译时嵌入 dist/，必须先于 cargo build）
Write-Host ""
Write-Host "[1/2] 构建前端..." -ForegroundColor Yellow
& npm run build
if ($LASTEXITCODE -ne 0) { throw "前端构建失败" }

# 2. 构建后端（无 MSVC，用 .tools zig + xwin 工具链）
Write-Host ""
Write-Host "[2/2] 构建后端 (cargo build --$profile)..." -ForegroundColor Yellow

$toolsDir = Join-Path $root ".tools"
if (Test-Path (Join-Path $toolsDir "zigwrap")) {
    Write-Host "  检测到 .tools 工具链，使用 zig + xwin 编译..." -ForegroundColor Gray
    $env:CC = Join-Path $toolsDir "zigwrap/zigcc.exe"
    $env:CXX = Join-Path $toolsDir "zigwrap/zigcxx.exe"
    $env:AR = Join-Path $toolsDir "zigwrap/ziglib.exe"
    $env:RC = Join-Path $toolsDir "zigwrap/zigrc.exe"
    $env:ZIG_GLOBAL_CACHE_DIR = Join-Path $toolsDir "zig-cache"
    $env:ZIG_LOCAL_CACHE_DIR = Join-Path $toolsDir "zig-cache/local"
    # 固定 rustup 工具链（rustup proxy 未设 RUSTUP_TOOLCHAIN 时 cargo 会静默 no-op）
    $env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-msvc"
    # 把 cargo.exe 加入 PATH（rustup 默认安装位置）
    $rustBin = Join-Path $env:USERPROFILE ".rustup/toolchains/stable-x86_64-pc-windows-msvc/bin"
    if (Test-Path $rustBin) {
        $env:PATH = "$rustBin;$env:PATH"
    }
}

Push-Location (Join-Path $root "src-tauri")
$cargoArgs = @("build")
if (-not $Debug) { $cargoArgs += "--release" }
& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { throw "后端构建失败" }
Pop-Location

$targetDir = if ($Debug) { "debug" } else { "release" }
$exe = Join-Path $root "src-tauri/target/$targetDir/pixsweep.exe"
Write-Host ""
Write-Host "构建完成: $exe" -ForegroundColor Green
