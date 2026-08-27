# 生成测试样本（多格式 + 重复/相似变体）
# 用 .NET System.Drawing 直接生成，无需任何依赖
# 输出：本脚本所在目录（test_assets/）

param([string]$OutDir = $PSScriptRoot)

Add-Type -AssemblyName System.Drawing

# 工具函数
function Fill-Color([System.Drawing.Bitmap]$img, [System.Drawing.Color]$c) {
    $g2 = [System.Drawing.Graphics]::FromImage($img)
    $g2.FillRectangle((New-Object System.Drawing.SolidBrush $c), 0, 0, $img.Width, $img.Height)
    $g2.Dispose()
}

function Draw-Gradient([System.Drawing.Bitmap]$img, [System.Drawing.Color]$c1, [System.Drawing.Color]$c2) {
    $g2 = [System.Drawing.Graphics]::FromImage($img)
    $rect = New-Object System.Drawing.Rectangle 0, 0, $img.Width, $img.Height
    $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $c1, $c2, [System.Drawing.Drawing2D.LinearGradientMode]::Vertical)
    $g2.FillRectangle($brush, $rect)
    $brush.Dispose()
    $g2.Dispose()
}

function Draw-Circles([System.Drawing.Bitmap]$img, [System.Drawing.Color]$c, [int]$count, [int]$seed = 42) {
    $g2 = [System.Drawing.Graphics]::FromImage($img)
    $rng = New-Object System.Random $seed
    $brush = New-Object System.Drawing.SolidBrush $c
    for ($i = 0; $i -lt $count; $i++) {
        $x = $rng.Next(0, $img.Width - 30)
        $y = $rng.Next(0, $img.Height - 30)
        $r = $rng.Next(8, 30)
        $g2.FillEllipse($brush, $x, $y, $r, $r)
    }
    $brush.Dispose()
    $g2.Dispose()
}

function Draw-Grid([System.Drawing.Bitmap]$img) {
    $g2 = [System.Drawing.Graphics]::FromImage($img)
    for ($y = 0; $y -lt 256; $y += 32) {
        for ($x = 0; $x -lt 256; $x += 32) {
            $isDark = ((($x + $y) / 32) % 2) -eq 0
            $c = if ($isDark) { [System.Drawing.Color]::Black } else { [System.Drawing.Color]::White }
            $brush = New-Object System.Drawing.SolidBrush $c
            $g2.FillRectangle($brush, $x, $y, 32, 32)
            $brush.Dispose()
        }
    }
    $g2.Dispose()
}

function Draw-Concentric([System.Drawing.Bitmap]$img, [System.Drawing.Color]$c1, [System.Drawing.Color]$c2) {
    $g2 = [System.Drawing.Graphics]::FromImage($img)
    $rect = New-Object System.Drawing.Rectangle 0, 0, $img.Width, $img.Height
    $brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $c1, $c2, [System.Drawing.Drawing2D.LinearGradientMode]::Diagonal)
    $g2.FillRectangle($brush, $rect)
    $brush.Dispose()
    for ($r = 20; $r -lt 128; $r += 16) {
        $g2.DrawEllipse([System.Drawing.Pens]::White, (128 - $r), (128 - $r), ($r * 2), ($r * 2))
    }
    $g2.Dispose()
}

function Draw-Noise([System.Drawing.Bitmap]$img, [int]$seed) {
    $rng = New-Object System.Random $seed
    for ($y = 0; $y -lt 256; $y += 2) {
        for ($x = 0; $x -lt 256; $x += 2) {
            $v = $rng.Next(0, 256)
            $c = [System.Drawing.Color]::FromArgb(255, $v, $v, $v)
            $img.SetPixel($x, $y, $c)
        }
    }
}

function Save-Img([System.Drawing.Bitmap]$img, [string]$path, [System.Drawing.Imaging.ImageFormat]$fmt, [int]$quality = 80) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $clone = New-Object System.Drawing.Bitmap $img  # 拷贝一份避免 Save 后 Dispose 影响
    if ($fmt -eq [System.Drawing.Imaging.ImageFormat]::Jpeg) {
        $codecs = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders()
        $codec = $codecs | Where-Object { $_.MimeType -eq 'image/jpeg' }
        $params = New-Object System.Drawing.Imaging.EncoderParameters 1
        $param = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, [int64]$quality)
        $params.Param[0] = $param
        $clone.Save($path, $codec, $params)
    } else {
        $clone.Save($path, $fmt)
    }
    $clone.Dispose()
}

# 清空旧目录
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Write-Host "=== Generating test samples ===" -ForegroundColor Cyan
Write-Host "Output: $OutDir"

$palette = New-Object System.Drawing.Color[] 6
$palette[0] = [System.Drawing.Color]::FromArgb(255, 12, 120, 200)
$palette[1] = [System.Drawing.Color]::FromArgb(255, 220, 60, 60)
$palette[2] = [System.Drawing.Color]::FromArgb(255, 60, 200, 80)
$palette[3] = [System.Drawing.Color]::FromArgb(255, 240, 180, 30)
$palette[4] = [System.Drawing.Color]::FromArgb(255, 150, 70, 220)
$palette[5] = [System.Drawing.Color]::FromArgb(255, 30, 200, 200)

# === 1. sunset（3 格式重复：PNG/JPG/BMP）===
$img1 = New-Object System.Drawing.Bitmap 256, 256
Draw-Gradient $img1 $palette[0] $palette[3]
Draw-Circles $img1 $palette[1] 5
Save-Img $img1 (Join-Path $OutDir "sunset.png") ([System.Drawing.Imaging.ImageFormat]::Png)
Save-Img $img1 (Join-Path $OutDir "sunset.jpg") ([System.Drawing.Imaging.ImageFormat]::Jpeg) 75
Save-Img $img1 (Join-Path $OutDir "sunset.bmp") ([System.Drawing.Imaging.ImageFormat]::Bmp)
$img1.Dispose()

# === 2. landscape（2 格式近似重复：PNG/JPG）===
$img2 = New-Object System.Drawing.Bitmap 256, 256
Draw-Gradient $img2 ([System.Drawing.Color]::FromArgb(255, 14, 130, 210)) ([System.Drawing.Color]::FromArgb(255, 250, 190, 40))
Draw-Circles $img2 ([System.Drawing.Color]::FromArgb(255, 230, 70, 70)) 5
Save-Img $img2 (Join-Path $OutDir "landscape.png") ([System.Drawing.Imaging.ImageFormat]::Png)
Save-Img $img2 (Join-Path $OutDir "landscape.jpg") ([System.Drawing.Imaging.ImageFormat]::Jpeg) 75
$img2.Dispose()

# === 3. circles（2 格式：PNG/GIF）===
$img3 = New-Object System.Drawing.Bitmap 256, 256
Draw-Concentric $img3 $palette[4] $palette[5]
Save-Img $img3 (Join-Path $OutDir "circles.png") ([System.Drawing.Imaging.ImageFormat]::Png)
Save-Img $img3 (Join-Path $OutDir "circles.gif") ([System.Drawing.Imaging.ImageFormat]::Gif)
$img3.Dispose()

# === 4. checker（2 格式：PNG/TIFF）===
$img4 = New-Object System.Drawing.Bitmap 256, 256
Draw-Grid $img4
Save-Img $img4 (Join-Path $OutDir "checker.png") ([System.Drawing.Imaging.ImageFormat]::Png)
Save-Img $img4 (Join-Path $OutDir "checker.tif") ([System.Drawing.Imaging.ImageFormat]::Tiff)
$img4.Dispose()

# === 5. noise（单张 BMP）===
$img5 = New-Object System.Drawing.Bitmap 256, 256
Draw-Noise $img5 7
Save-Img $img5 (Join-Path $OutDir "noise.bmp") ([System.Drawing.Imaging.ImageFormat]::Bmp)
$img5.Dispose()

# === 6. noise2（另一个随机种子 PNG）===
$img6 = New-Object System.Drawing.Bitmap 256, 256
Draw-Noise $img6 99
Save-Img $img6 (Join-Path $OutDir "noise2.png") ([System.Drawing.Imaging.ImageFormat]::Png)
$img6.Dispose()

# === 7. solid（纯色 PNG）===
$img7 = New-Object System.Drawing.Bitmap 256, 256
Fill-Color $img7 $palette[2]
Save-Img $img7 (Join-Path $OutDir "solid.png") ([System.Drawing.Imaging.ImageFormat]::Png)
$img7.Dispose()

# === 8. landscape_v2（轻微变化 JPG — 类似 landscape）===
$img8 = New-Object System.Drawing.Bitmap 256, 256
Draw-Gradient $img8 ([System.Drawing.Color]::FromArgb(255, 18, 140, 220)) ([System.Drawing.Color]::FromArgb(255, 245, 185, 35))
Draw-Circles $img8 $palette[1] 6
Save-Img $img8 (Join-Path $OutDir "landscape_v2.jpg") ([System.Drawing.Imaging.ImageFormat]::Jpeg) 70
$img8.Dispose()

# === 9. sunset_v2（细节丰富版 JPG — 类似 sunset）===
$img9 = New-Object System.Drawing.Bitmap 256, 256
Draw-Gradient $img9 $palette[0] $palette[3]
Draw-Circles $img9 $palette[1] 15
Save-Img $img9 (Join-Path $OutDir "sunset_v2.jpg") ([System.Drawing.Imaging.ImageFormat]::Jpeg) 80
$img9.Dispose()

# 显示结果
Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Get-ChildItem $OutDir -File | Format-Table Name, Length -AutoSize | Out-String | Write-Host

$totalSize = (Get-ChildItem $OutDir -File | Measure-Object Length -Sum).Sum
Write-Host ("Total: {0:N2} MB" -f ($totalSize / 1MB)) -ForegroundColor Green
