//! 目录扫描：递归遍历文件夹，识别支持的图片格式。

use crate::types::ImageInfo;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// 支持的图片扩展名（小写）。
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "heic", "heif",
];

/// 支持的相机 RAW 扩展名（小写），与 [`crate::image_io::RAW_EXTENSIONS`] 保持一致。
/// RAW 走 rawler 解码（机内嵌预览优先，见 image_io::load_raw_oriented）。
const SUPPORTED_RAW_EXTENSIONS: &[&str] = &[
    "rw2", "nef", "nrw", "arw", "srw", "cr2", "cr3", "crw", "raf", "orf", "pef",
    "ptx", "dng", "raw", "rwl", "x3f", "3fr", "erf", "mrw", "iiq", "gpr", "kdc", "dcr",
];

/// 判断路径是否为支持的图片文件。
pub fn is_supported_image(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext = ext.to_ascii_lowercase();
    SUPPORTED_EXTENSIONS.contains(&ext.as_str())
        || SUPPORTED_RAW_EXTENSIONS.contains(&ext.as_str())
}

/// 生成文件指纹：基于路径 + 大小 + 修改时间。用于增量扫描判定文件是否变化。
fn file_fingerprint(path: &Path, size: u64, modified: u64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(&size.to_le_bytes());
    hasher.update(&modified.to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

/// 扫描一个文件夹，返回所有支持的图片文件信息。
pub fn scan_folder(folder: &Path) -> Vec<ImageInfo> {
    let mut result = Vec::new();

    for entry in WalkDir::new(folder).follow_links(false).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !entry.file_type().is_file() || !is_supported_image(path) {
            continue;
        }

        let Ok(meta) = std::fs::metadata(path) else {
            continue;
        };

        let size = meta.len();
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let file_hash = file_fingerprint(path, size, modified);

        result.push(ImageInfo {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            path: path.to_string_lossy().to_string(),
            size,
            modified,
            width: 0,
            height: 0,
            format: path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase(),
            file_hash,
        });
    }

    result
}

/// 扫描多个文件夹，去重后返回。
pub fn scan_folders(folders: &[PathBuf]) -> Vec<ImageInfo> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for folder in folders {
        for img in scan_folder(folder) {
            // 以绝对路径去重（同一文件被多个文件夹覆盖时只保留一次）
            if seen.insert(img.path.clone()) {
                result.push(img);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_extensions_cover_common_and_raw() {
        for p in [
            "a.jpg", "a.jpeg", "a.png", "a.webp", "a.heic",
            // 主流品牌 RAW
            "a.rw2", "a.nef", "a.arw", "a.cr2", "a.cr3", "a.raf", "a.orf",
            "a.dng", "a.raw", "a.iiq",
            // 大小写不敏感
            "a.NEF", "a.Rw2",
        ] {
            assert!(is_supported_image(Path::new(p)), "应支持: {p}");
        }
        for p in ["a.txt", "a.mp4", "a.exe", "a.tifff", "a"] {
            assert!(!is_supported_image(Path::new(p)), "不应支持: {p}");
        }
    }
}
