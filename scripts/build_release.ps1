# PixSweep 打包脚本：构建并打包为可离线运行的 zip
# 产物：dist-package/PixSweep-v{version}.zip，解压即用
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1

param(
    [string]$Version = "",
    [string]$OutputDir = "dist-package"
)

$ErrorActionPreference = "Stop"

# 切换到脚本所在目录的父目录（项目根）
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Set-Location ".."
$root = (Get-Location).Path

Write-Host "=== PixSweep 打包脚本 ===" -ForegroundColor Cyan
Write-Host "项目根: $root"

# 1. 读取版本号（默认从 tauri.conf.json 读取，与 Cargo.toml / package.json 保持一致）
if ($Version -eq "") {
    $confPath = Join-Path $root "src-tauri/tauri.conf.json"
    $conf = Get-Content $confPath -Raw | ConvertFrom-Json
    $Version = $conf.version
    if (-not $Version) { $Version = "0.1.0" }
}
Write-Host "版本号: $Version"

# 2. 构建前端
Write-Host ""
Write-Host "[1/5] 构建前端..." -ForegroundColor Yellow
& npm run build
if ($LASTEXITCODE -ne 0) { throw "前端构建失败" }

# 3. 构建后端（release）
Write-Host ""
Write-Host "[2/5] 构建后端 (cargo build --release)..." -ForegroundColor Yellow

# 检测是否有 .tools 工具链（无 MSVC 环境）
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
& cargo build --release
if ($LASTEXITCODE -ne 0) { throw "后端构建失败" }
Pop-Location

$exe = Join-Path $root "src-tauri/target/release/pixsweep.exe"
if (-not (Test-Path $exe)) { throw "未找到编译产物: $exe" }

# 4. 准备打包目录
Write-Host ""
Write-Host "[3/5] 准备打包目录..." -ForegroundColor Yellow
$pkgName = "PixSweep-v$Version"
$pkgDir = Join-Path $root (Join-Path $OutputDir $pkgName)
if (Test-Path $pkgDir) { Remove-Item $pkgDir -Recurse -Force }
New-Item -ItemType Directory -Path $pkgDir -Force | Out-Null

# 主程序
Copy-Item $exe (Join-Path $pkgDir "PixSweep.exe")

# ONNX Runtime + DirectML DLL
# - DirectML.dll + providers_shared：所有 GPU 通用，DirectML EP 兜底
# - onnxruntime_providers_cuda.dll：装过 CUDA 运行时（cudart/cudnn）的机器启用 CUDA EP，
#   其余机器加载失败会自动回退 DirectML/CPU（优雅降级，不报错）
$targetRelease = Join-Path $root "src-tauri/target/release"
$dlls = @("DirectML.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")
foreach ($dll in $dlls) {
    $src = Join-Path $targetRelease $dll
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $pkgDir $dll)
        Write-Host "  复制 $dll" -ForegroundColor Gray
    }
}

# AI 模型文件（只打包代码实际引用的模型，避免冗余增大产物）
# 引用见 src-tauri/src/ai/engine.rs：CLIP_MODEL / CLIP_IQA_MODEL / NIMA_TECH_MODEL / AESTHETIC_WEIGHTS
$modelDir = Join-Path $root "src-tauri/models"
if (-not (Test-Path $modelDir)) { $modelDir = Join-Path $root "models" }
if (Test-Path $modelDir) {
    $destModels = Join-Path $pkgDir "models"
    New-Item -ItemType Directory -Path $destModels -Force | Out-Null
    $neededModels = @(
        "topiq_nr.onnx",              # TOPIQ_NR_MODEL：主技术质量评分图结构（ResNet50，KonIQ-10k，动态 batch 单文件）
        "topiq_iaa_res50.onnx",       # TOPIQ_IAA_MODEL：主美学评分（ResNet50，AVA，动态 batch 单文件）
        "nima-technical.onnx",        # NIMA_TECH_MODEL：技术评分二级后备
        "topiq_nr_face.onnx",         # TOPIQ_NR_FACE_MODEL：人脸专评（有人脸档最高权重）
        "topiq_nr_face.onnx.data"     # TOPIQ_NR_FACE_MODEL 外部权重（与 .onnx 配对）
    )
    foreach ($m in $neededModels) {
        $src = Join-Path $modelDir $m
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $destModels $m)
            Write-Host "  复制模型: $m" -ForegroundColor Gray
        } else {
            Write-Host "  警告: 缺少模型文件 $m" -ForegroundColor Yellow
        }
    }
    # 人像专评/场景/闭眼子目录模型（引擎从 models/<子目录>/ 加载，见 engine.rs）
    # insightface：仅 det_10g（load() 只要求这个；2d106det/genderage 未使用，不打包）
    $subModels = @{
        "insightface" = @("det_10g.onnx")
        "scene"       = @("mobilenet_v3_large.onnx", "mobilenet_v3_large.data", "labels.txt")
        "eye"         = @("ocec_l.onnx", "face_landmarker.onnx")  # face_landmarker：垂目闭眼脸网格（可选信号，缺则仅 OCEC）
    }
    foreach ($sub in $subModels.Keys) {
        $srcSub = Join-Path $modelDir $sub
        if (-not (Test-Path $srcSub)) { continue }
        $destSub = Join-Path $destModels $sub
        New-Item -ItemType Directory -Path $destSub -Force | Out-Null
        foreach ($m in $subModels[$sub]) {
            $src = Join-Path $srcSub $m
            if (Test-Path $src) {
                Copy-Item $src (Join-Path $destSub $m)
                Write-Host "  复制模型: $sub/$m" -ForegroundColor Gray
            } else {
                Write-Host "  警告: 缺少模型文件 $sub/$m" -ForegroundColor Yellow
            }
        }
    }
    # 提示被排除的未引用模型
    $skipped = Get-ChildItem -Path $modelDir -File | Where-Object {
        ($_.Extension -in @(".onnx", ".bin")) -and ($neededModels -notcontains $_.Name)
    }
    foreach ($s in $skipped) {
        Write-Host "  跳过未引用模型: $($s.Name) ($('{0:N1}' -f ($s.Length / 1MB)) MB)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  警告: 未找到模型目录，AI 评分功能将不可用" -ForegroundColor Red
}

# 5. 生成 zip（优先 7-Zip 最大压缩 mx=9，回退 Compress-Archive Optimal）
Write-Host ""
Write-Host "[4/5] 生成 zip 包..." -ForegroundColor Yellow
$zipDir = Join-Path $root $OutputDir
$zipPath = Join-Path $zipDir "${pkgName}.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$sevenZip = Join-Path $toolsDir "7zip/7za.exe"
if (Test-Path $sevenZip) {
    Write-Host "  使用 7-Zip (zip/mx=7/mmt，平衡压缩率与打包速度): $sevenZip" -ForegroundColor Gray
    Push-Location $pkgDir
    & $sevenZip a -tzip -mx=7 -mmt=on -bd -y $zipPath "*"
    $exitCode = $LASTEXITCODE
    Pop-Location
    if ($exitCode -ne 0) { throw "7-Zip 打包失败 (exit=$exitCode)" }
} else {
    Write-Host "  未找到 7-Zip (.tools/7zip/7za.exe)，回退 Compress-Archive Optimal" -ForegroundColor Yellow
    Compress-Archive -Path "$pkgDir/*" -DestinationPath $zipPath -CompressionLevel Optimal
}

# 6. 验证
Write-Host ""
Write-Host "[5/5] 验证打包结果..." -ForegroundColor Yellow
$zipSize = (Get-Item $zipPath).Length / 1MB
Write-Host "  zip 文件: $zipPath" -ForegroundColor Green
Write-Host "  大小: $('{0:N1}' -f $zipSize) MB" -ForegroundColor Green
Write-Host "  内容:" -ForegroundColor Green
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
foreach ($entry in $zip.Entries) {
    if ($entry.Length -eq 0) { continue }
    $origMB = '{0:N1}' -f ($entry.Length / 1MB)
    $compMB = '{0:N1}' -f ($entry.CompressedLength / 1MB)
    $ratio = '{0:P0}' -f ($entry.CompressedLength / $entry.Length)
    Write-Host "    $($entry.FullName)  $origMB MB -> $compMB MB ($ratio)" -ForegroundColor Gray
}
$zip.Dispose()

Write-Host ""
Write-Host "=== 打包完成 ===" -ForegroundColor Cyan
Write-Host "分发方式：将 zip 发给用户，解压后双击 PixSweep.exe 即可运行" -ForegroundColor Green
