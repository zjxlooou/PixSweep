//! PixSweep 桌面应用入口（bin 目标）。
//!
//! 仅做两件事：声明 Windows 发布版隐藏控制台窗口，然后委托给
//! [`pixsweep_lib::run`] 启动 Tauri 应用。所有业务逻辑都在 lib crate 中。

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    pixsweep_lib::run()
}
