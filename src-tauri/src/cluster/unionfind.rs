//! Union-Find（并查集）数据结构，用于连通分量聚类。

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // 路径压缩
        }
        self.parent[x]
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // 按秩合并
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }

    /// 返回所有连通分量（每个分量是一个索引列表）。
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut map: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            map.entry(root).or_default().push(i);
        }
        let mut comps: Vec<Vec<usize>> = map.into_values().collect();
        // 稳定排序：HashMap 遍历顺序随机会导致组/照片每次扫描顺序不同。
        // 组内按索引升序、组之间按组内最小索引排序，保证跨扫描结果稳定。
        for c in comps.iter_mut() {
            c.sort_unstable();
        }
        comps.sort_by_key(|c| c[0]);
        comps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_union_find() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        let mut comps = uf.components();
        for c in comps.iter_mut() {
            c.sort();
        }
        comps.sort();
        assert_eq!(comps, vec![vec![0, 1, 2], vec![3, 4]]);
    }
}
