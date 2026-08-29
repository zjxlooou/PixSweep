//! 硬件探测与并发参数动态化（2026-08-29）。
//!
//! 扫描前置的硬件画像：CPU 逻辑核心数、系统内存、独显显存（DXGI）。
//! 据此动态确定两组此前硬编码的并发参数：
//! - **解码/重活线程数**（`heavy_pool` + 解码信号量）：按核心数与内存分档；
//! - **SCRFD 会话副本数**：按显存分档——实测 6GB 卡上 4 副本会顶爆显存使后续
//!   模型减速 6×（见 docs/GPU_PERF_PLAN.md），副本数必须随显存收缩。
//!
//! 全部结果进程内缓存（`OnceLock`），启动时打日志便于核对。

use std::sync::OnceLock;

/// 硬件画像（探测失败的维度用保守默认值填充）。
#[derive(Debug, Clone, Copy)]
pub struct HardwareProfile {
    /// CPU 逻辑核心数
    pub logical_cores: usize,
    /// 系统内存（GiB，向下取整）
    pub total_ram_gb: u64,
    /// 独显显存（GiB，向下取整）；未知为 0
    pub vram_gb: u64,
    /// 重活/解码并发线程数
    pub decode_threads: usize,
    /// SCRFD 会话副本数
    pub det_replicas: usize,
}

/// 解码/重活并发线程数（核心数 × 内存双约束）。
///
/// - 核心维度：`clamp(逻辑核 / 2, 2, 8)`——解码是内存带宽型，半数核已饱和；
/// - 内存维度：每线程峰值解码缓冲 100~200MB，小内存机器封顶。
pub fn decode_threads() -> usize {
    profile().decode_threads
}

/// SCRFD 会话副本数（显存分档）。
///
/// 实测锚点：6GB 卡 4 副本 → 显存 91%+、后续模型减速 6×，**2 副本最优**；
/// 更高档位为外推估计，未实测。
pub fn det_replicas() -> usize {
    profile().det_replicas
}

fn profile() -> &'static HardwareProfile {
    static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();
    PROFILE.get_or_init(detect)
}

fn detect() -> HardwareProfile {
    let logical_cores = logical_cores();
    let total_ram_gb = total_ram_gb();
    let vram_gb = vram_gb();

    let decode_threads = decode_threads_for(logical_cores, total_ram_gb);
    let det_replicas = det_replicas_for(vram_gb, total_ram_gb);

    log::info!(
        "[硬件] 逻辑核 {} | 内存 {}GB | 显存 {}GB -> 重活线程 {} | SCRFD 副本 {}",
        logical_cores,
        total_ram_gb,
        if vram_gb == 0 { "未知".to_string() } else { vram_gb.to_string() },
        decode_threads,
        det_replicas
    );

    HardwareProfile { logical_cores, total_ram_gb, vram_gb, decode_threads, det_replicas }
}

/// 纯函数：核心数 + 内存（GB）→ 重活线程数。便于单测。
fn decode_threads_for(logical_cores: usize, total_ram_gb: u64) -> usize {
    let by_core = (logical_cores / 2).clamp(2, 8);
    // 16GB 机器探测值为 15.x（GiB 向下取整），阈值取 15
    let by_ram = if total_ram_gb >= 15 {
        8 // 16GB 实测 6 线程内存无压力，跟核心数走即可
    } else if total_ram_gb >= 8 {
        4
    } else {
        2
    };
    by_core.min(by_ram)
}

/// 纯函数：显存（GB）+ 内存（GB）→ SCRFD 副本数。便于单测。
fn det_replicas_for(vram_gb: u64, total_ram_gb: u64) -> usize {
    // 内存过小（<8GB）时系统本身就紧张，副本降至 1
    if total_ram_gb < 8 {
        return 1;
    }
    match vram_gb {
        0 => 2, // 显存未知（探测失败）→ 用已验证的保守默认
        v if v >= 12 => 4,
        v if v >= 8 => 3,
        v if v >= 5 => 2,
        _ => 1,
    }
}

fn logical_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

#[cfg(windows)]
fn total_ram_gb() -> u64 {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut ms: MEMORYSTATUSEX = std::mem::zeroed();
        ms.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut ms).is_ok() {
            return ms.ullTotalPhys / (1024 * 1024 * 1024);
        }
    }
    16 // 探测失败按 16GB 保守处理
}

#[cfg(not(windows))]
fn total_ram_gb() -> u64 {
    16
}

/// 独显显存（取最大适配器的 DedicatedVideoMemory，DXGI）。
#[cfg(windows)]
fn vram_gb() -> u64 {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
        Ok(f) => f,
        Err(_) => return 0,
    };
    let mut best: u64 = 0;
    for i in 0..8u32 {
        let adapter = match (unsafe { factory.EnumAdapters1(i) }).ok() {
            Some(a) => a,
            None => break,
        };
        let mut desc = Default::default();
        if (unsafe { adapter.GetDesc1(&mut desc) }).is_ok() {
            // 排除微软基本渲染（Software Adapter 标志位）
            if (desc.Flags & 2) == 0 {
                best = best.max(desc.DedicatedVideoMemory as u64);
            }
        }
    }
    best / (1024 * 1024 * 1024)
}

#[cfg(not(windows))]
fn vram_gb() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_threads_mapping() {
        // 核心维度
        assert_eq!(decode_threads_for(4, 32), 2);
        assert_eq!(decode_threads_for(8, 32), 4);
        assert_eq!(decode_threads_for(16, 32), 8);
        assert_eq!(decode_threads_for(32, 32), 8); // 上限
        // 内存封顶
        assert_eq!(decode_threads_for(16, 16), 8);
        assert_eq!(decode_threads_for(12, 15), 6); // 本机：12 核 16GB(报 15) -> 6
        assert_eq!(decode_threads_for(16, 8), 4);
        assert_eq!(decode_threads_for(16, 4), 2);
    }

    #[test]
    fn det_replicas_mapping() {
        // 显存锚点：6GB 卡实测 2 副本最优
        assert_eq!(det_replicas_for(6, 32), 2);
        assert_eq!(det_replicas_for(5, 32), 2);
        assert_eq!(det_replicas_for(4, 32), 1);
        assert_eq!(det_replicas_for(8, 32), 3);
        assert_eq!(det_replicas_for(12, 32), 4);
        assert_eq!(det_replicas_for(24, 32), 4);
        assert_eq!(det_replicas_for(0, 32), 2); // 未知 → 保守默认
        assert_eq!(det_replicas_for(24, 4), 1); // 内存过小 → 1
    }
}
