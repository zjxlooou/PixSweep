//! 相似度计算与聚类入口。

use super::unionfind::UnionFind;
use crate::hashing::phash;

/// 判断哈希是否"无特征"（几乎无梯度），如纯色/平滑渐变图。
/// 这类图片的 dhash 不可靠，应跳过配对，避免误判为相似。
fn is_featureless(hash: u64) -> bool {
    let c = hash.count_ones();
    c < 8 || c > 56
}

/// ahash 双哈希校验的最低相似度。
/// dhash 对平滑渐变图不敏感（相邻像素明暗关系几乎同向），两张内容完全不同的
/// 渐变图 dhash 也可能 >92%；ahash 比较像素与全局均值，能正确区分。
/// 实测：真重复 100%，真相似(重编码) 100%，渐变误判组 58% —— 0.80 有足够余量。
pub const AHASH_SIMILARITY_THRESHOLD: f32 = 0.80;

/// 基于感知哈希聚类：dhash 与 ahash **双条件**同时满足才归为一组。
///
/// 返回每个分组的图片索引列表（组内成员数 >= 2）。
pub fn cluster_by_hash(hashes: &[u64], ahashs: &[u64], threshold: f32) -> Vec<Vec<usize>> {
    let n = hashes.len();
    if n < 2 {
        return Vec::new();
    }
    debug_assert_eq!(hashes.len(), ahashs.len(), "dhash 与 ahash 数组长度必须一致");

    // 预计算 featureless，避免内层循环重复 popcount
    let featureless: Vec<bool> = hashes.iter().map(|h| is_featureless(*h)).collect();

    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            // 跳过两张都是"无特征"（平滑）图片的配对，避免误判
            if featureless[i] && featureless[j] {
                continue;
            }
            // 双哈希校验：dhash（对压缩/重编码稳健）+ ahash（对渐变图敏感）
            if phash::dhash_similarity(hashes[i], hashes[j]) >= threshold
                && phash::ahash_similarity(ahashs[i], ahashs[j]) >= AHASH_SIMILARITY_THRESHOLD
            {
                uf.union(i, j);
            }
        }
    }

    uf.components()
        .into_iter()
        .filter(|g| g.len() >= 2)
        .collect()
}

/// 计算一组图片之间的平均相似度（基于哈希）。
pub fn average_hash_similarity(hashes: &[u64], group: &[usize]) -> f32 {
    if group.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0f32;
    let mut count = 0u32;
    for a in 0..group.len() {
        for b in (a + 1)..group.len() {
            total += phash::dhash_similarity(hashes[group[a]], hashes[group[b]]);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_by_hash_groups_similar() {
        // 相同哈希聚为一组，不同哈希各自独立
        let hashes = vec![0xAAAAu64, 0xAAAA, 0x5555];
        let ahashs = vec![0xFF00u64, 0xFF00, 0x00FF];
        let groups = cluster_by_hash(&hashes, &ahashs, 0.9);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn cluster_by_hash_rejects_gradient_mismatch() {
        // 模拟"渐变图误判"：dhash 很接近（>0.92）但 ahash 差异大（<0.80）
        // → 双哈希应拒绝分组
        let hashes = vec![0xF4C0D8F8_F0E0F0F0u64, 0xF0E0D8D8_F0F0F0E0u64];
        let ahashs = vec![0x00000000_00000000u64, 0xFFFF0000_00000000u64];
        let groups = cluster_by_hash(&hashes, &ahashs, 0.92);
        assert_eq!(groups.len(), 0, "渐变图误判应被 ahash 双哈希拒绝");
    }
}
