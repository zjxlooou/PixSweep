//! 感知哈希（Perceptual Hash）。
//!
//! 提供两种快速的图像指纹算法，用于检测「视觉上相同/高度相似」的图片：
//! - [`dhash`] 差异哈希（对轻微亮度/压缩变化更稳健）
//! - [`ahash`] 平均哈希
//!
//! 感知哈希是轻量级预筛手段，可在不加载 AI 模型的情况下快速识别精确重复；
//! 双哈希（dhash + ahash）联合校验过滤渐变图误判（见 `cluster::similarity`）。

use image::DynamicImage;

/// 计算灰度缩略图（缩放到 width x height）。
fn grayscale_thumbnail(img: &DynamicImage, width: u32, height: u32) -> Vec<u8> {
    let gray = img.to_luma8();
    let resized = image::imageops::resize(&gray, width, height, image::imageops::FilterType::Lanczos3);
    resized.into_raw()
}

/// 差异哈希（dhash）：缩放到 9x8，逐行比较相邻像素，得到 64-bit。
pub fn dhash(img: &DynamicImage) -> u64 {
    let (w, h) = (9u32, 8u32);
    let pixels = grayscale_thumbnail(img, w, h);
    let mut hash: u64 = 0;

    for row in 0..h {
        for col in 0..(w - 1) {
            let left = pixels[(row * w + col) as usize];
            let right = pixels[(row * w + col + 1) as usize];
            if left > right {
                hash |= 1 << (row * (w - 1) + col);
            }
        }
    }
    hash
}

/// 平均哈希（ahash）：缩放到 8x8，与平均灰度比较，得到 64-bit。
pub fn ahash(img: &DynamicImage) -> u64 {
    let (w, h) = (8u32, 8u32);
    let pixels = grayscale_thumbnail(img, w, h);

    let avg = pixels.iter().map(|&p| p as u64).sum::<u64>() / (w * h) as u64;

    let mut hash: u64 = 0;
    for (i, &p) in pixels.iter().enumerate() {
        if p as u64 > avg {
            hash |= 1 << i;
        }
    }
    hash
}

/// 汉明距离（两个哈希之间不同的位数）。
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// 将汉明距离归一化为相似度（0.0 ~ 1.0，1.0 表示完全相同）。
pub fn similarity_from_distance(distance: u32, bits: u32) -> f32 {
    1.0 - distance as f32 / bits as f32
}

/// 计算两个 dhash 的相似度。
pub fn dhash_similarity(a: u64, b: u64) -> f32 {
    similarity_from_distance(hamming_distance(a, b), 64)
}

/// 计算两个 ahash 的平均哈希相似度。
///
/// ahash 对"平滑渐变图"（纯色背景、灯光夜景等）更敏感——这类图片的
/// dhash 相邻像素明暗关系几乎同向，容易误判为相似；ahash 比较像素与
/// 全局均值的关系，能正确区分。与 [`dhash_similarity`] 配合做双哈希校验。
pub fn ahash_similarity(a: u64, b: u64) -> f32 {
    similarity_from_distance(hamming_distance(a, b), 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_have_identical_hash() {
        // 构造两张完全相同的小图
        let mut img1 = image::GrayImage::new(16, 16);
        let mut img2 = image::GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let v = ((x + y) % 255) as u8;
                img1.put_pixel(x, y, image::Luma([v]));
                img2.put_pixel(x, y, image::Luma([v]));
            }
        }
        let d1 = dhash(&DynamicImage::ImageLuma8(img1));
        let d2 = dhash(&DynamicImage::ImageLuma8(img2));
        assert_eq!(d1, d2);
        assert_eq!(hamming_distance(d1, d2), 0);
    }

    #[test]
    fn hamming_distance_works() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0b0000, 0b1111), 4);
        assert_eq!(hamming_distance(0b1010, 0b1010), 0);
    }
}
