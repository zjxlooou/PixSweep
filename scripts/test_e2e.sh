#!/usr/bin/env bash
# PixSweep 标准端到端测试（Git Bash / WSL 友好）
# 流程：复制样本 → 启动 app + MCP → 扫描 → 验证分组 →
#       设置功能（AI 评分/阈值/增量）→ 功能链路（删除/回收站/恢复/清空）→ 导出 → 清理
#
# 用法：bash scripts/test_e2e.sh
# 注意：不要 set -e，grep 无匹配时返回 1 会终止脚本

# 项目根：默认从脚本自身位置推导（Windows 风格路径）；也可用环境变量覆盖
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && { pwd -W 2>/dev/null || pwd; })}"
SAMPLE_DIR="$PROJECT_ROOT/test_assets"
# 用 Windows 风格 TEMP 路径（PixSweep 是 Windows 应用），从系统环境推导，不硬编码本机路径
if command -v cygpath >/dev/null 2>&1 && [ -n "${LOCALAPPDATA:-}" ]; then
    TEMP_ROOT="$(cygpath -m "$LOCALAPPDATA")/Temp/pixsweep_e2e_test"
else
    TEMP_ROOT="${TEMP_ROOT:-/tmp/pixsweep_e2e_test}"
fi
SCAN_DIR="$TEMP_ROOT/scan_input"
REPORT="$TEMP_ROOT/report.csv"
EXE="$PROJECT_ROOT/src-tauri/target/release/pixsweep.exe"
MCP_URL="http://127.0.0.1:18765/mcp"

echo "====== PixSweep Standard E2E Test ======"
echo "Project: $PROJECT_ROOT"
echo "Sample:  $SAMPLE_DIR"
echo ""

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
# 调用 MCP 工具，输出完整 JSON 响应
# - Connection: close 强制短连接（手写 HTTP server 对 keep-alive 复用有竞态，偶发响应错位）
# - curl 层 --retry-connrefused 处理连接被拒/瞬间失败（Windows 快速连续连接偶发）
# - 空响应/非 result 响应自动重试
mcp_call() {
    local tool="$1" args="$2" resp=""
    for attempt in 1 2 3; do
        resp=$(curl -s --max-time 60 --retry 2 --retry-connrefused --retry-delay 1 \
            -H "Connection: close" -X POST "$MCP_URL" \
            -H "Content-Type: application/json" \
            -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$tool\",\"arguments\":$args}}")
        if [ -n "$resp" ] && printf '%s' "$resp" | grep -q '"result"'; then
            break
        fi
        sleep 1
    done
    printf '%s' "$resp"
}

# 从 MCP 响应中提取 content[0].text（isError 时输出错误并返回 2）
mcp_text() {
    python -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print(f'INVALID_JSON: {e}')
    sys.exit(2)
r = d.get('result', {})
if r.get('isError'):
    print(f'TOOL_ERROR: {r.get(\"content\", [{}])[0].get(\"text\", \"\")}')
    sys.exit(2)
print(r.get('content', [{}])[0].get('text', ''))
"
}

# 杀 PixSweep（Git Bash 中 taskkill 用单斜杠即可，双斜杠 //F 会报无效参数）
kill_pixsweep() {
    taskkill /F /IM PixSweep.exe 2>/dev/null
    return 0
}
# 生成 JSON 数组参数
json_array() { python -c "import json,sys; print(json.dumps(sys.argv[1:]))" "$@"; }

# ---------------------------------------------------------------------------
# 阶段 0: 准备
# ---------------------------------------------------------------------------
[ -d "$SAMPLE_DIR" ] || { echo "ERROR: Sample dir not found: $SAMPLE_DIR"; exit 1; }
rm -rf "$TEMP_ROOT" && mkdir -p "$SCAN_DIR"
# ---------------------------------------------------------------------------
# 阶段 1: 复制样本（隔离测试，不污染原始样本）
# ---------------------------------------------------------------------------
echo "[1/8] Copy samples to isolated temp dir..."
cp "$SAMPLE_DIR"/*.png "$SAMPLE_DIR"/*.jpg "$SAMPLE_DIR"/*.bmp "$SAMPLE_DIR"/*.gif "$SAMPLE_DIR"/*.tif "$SCAN_DIR/" 2>/dev/null
SAMPLE_FILES=$(find "$SCAN_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sb "$SCAN_DIR" 2>/dev/null | cut -f1 || echo 0)
echo "  Copied $SAMPLE_FILES files ($((TOTAL_SIZE / 1024)) KB)"

# ---------------------------------------------------------------------------
# 阶段 2: 启动 PixSweep + MCP
# ---------------------------------------------------------------------------
echo ""
echo "[2/8] Launch PixSweep with MCP..."
kill_pixsweep; sleep 1
sleep 1
nohup "$EXE" --mcp > /dev/null 2>&1 &
sleep 5

READY=0
for i in $(seq 1 15); do
    RESP=$(curl -s --max-time 2 -X POST "$MCP_URL" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' 2>/dev/null)
    if echo "$RESP" | grep -q '"pixsweep"'; then
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    kill_pixsweep; sleep 1
    echo "ERROR: MCP server failed to start within 15s"
    exit 1
fi
echo "  MCP ready"

# 清空回收站（保证测试起点干净，避免上次残留影响计数）
mcp_call clear_trash "{}" | mcp_text > /dev/null 2>&1
echo "  Trash cleared (clean start)"

# ---------------------------------------------------------------------------
# 阶段 3: 扫描
# ---------------------------------------------------------------------------
echo ""
echo "[3/8] Run scan via MCP..."
SCAN_BODY=$(printf '{"folders":["%s"],"incremental":false}' "$SCAN_DIR")
SCAN_RESP=$(mcp_call start_scan "$SCAN_BODY")
SCAN_JSON=$(printf '%s' "$SCAN_RESP" | mcp_text) || { echo "  Scan failed: $SCAN_JSON"; exit 1; }

# 提取扫描摘要指标
SCAN_METRICS=$(printf '%s' "$SCAN_JSON" | python -c "
import json, sys
text = sys.stdin.read()
# text 以摘要开头（'扫描完成：...'），找到 JSON 对象起点
idx = text.find('{')
if idx < 0:
    raise SystemExit('NO_JSON_IN_TEXT')
scan = json.loads(text[idx:])
total = scan.get('total_images', 0)
groups = scan.get('groups', [])
group_count = sum(1 for g in groups if len(g.get('images', [])) > 1)
# 找到 sunset 组
sunset_group = None
for g in groups:
    names = [i['info']['name'] for i in g.get('images', [])]
    if 'sunset.png' in names:
        sunset_group = names
        break
print(f'TOTAL={total}')
print(f'GROUPS={group_count}')
print(f'SUNSET_GROUP={sunset_group}')
print(f'ALL_GROUPS={[[i[\"info\"][\"name\"] for i in g.get(\"images\", [])] for g in groups]}')
")
TOTAL=$(echo "$SCAN_METRICS" | awk -F= '/^TOTAL=/{print $2; exit}' | tr -d '\r')
N_GROUPS=$(echo "$SCAN_METRICS" | awk -F= '/^GROUPS=/{print $2; exit}' | tr -d '\r')
SUNSET_GROUP=$(echo "$SCAN_METRICS" | awk -F= '/^SUNSET_GROUP=/{ $1=""; sub(/^ /,""); print; exit }' | tr -d '\r')
echo "  Total images: $TOTAL"

# ---------------------------------------------------------------------------
# 阶段 4: 验证分组
# ---------------------------------------------------------------------------
echo ""
echo "[4/8] Verify grouping..."

PASS=1
if [ "$TOTAL" -ge 14 ]; then
    echo "  [PASS] All 14 sample images scanned (total=$TOTAL)"
else
    echo "  [FAIL] Only $TOTAL images found in scan"
    PASS=0
fi

if [ "$N_GROUPS" -ge 3 ]; then
    echo "  [PASS] Found >=3 groups (n=$N_GROUPS)"
else
    echo "  [FAIL] Found only $N_GROUPS groups (expected >=3)"
    PASS=0
fi

# 验证 sunset 跨格式识别（png + jpg + bmp 在同一组）
if echo "$SUNSET_GROUP" | grep -q "sunset.png" && echo "$SUNSET_GROUP" | grep -q "sunset.jpg" && echo "$SUNSET_GROUP" | grep -q "sunset.bmp"; then
    echo "  [PASS] Cross-format detection: sunset.png+jpg+bmp in same group"
else
    echo "  [FAIL] sunset cross-format not in same group: $SUNSET_GROUP"
    PASS=0
fi

# ---------------------------------------------------------------------------
# 阶段 5: 设置功能测试（AI 评分开关 / 相似度阈值 / 增量扫描）
# ---------------------------------------------------------------------------
echo ""
echo "[5/8] Settings test (AI scoring / threshold / incremental)..."

# --- 5.1 读取当前设置，验证字段完整 ---
echo "  5.1 get_settings fields present"
CUR_SETTINGS=$(mcp_call get_settings "{}" | mcp_text)
FIELDS_OK=$(printf '%s' "$CUR_SETTINGS" | python -c "
import json, sys
d = json.loads(sys.stdin.read())
req = ['similarity_threshold', 'ai_enabled', 'permanent_delete', 'incremental', 'mcp_enabled']
missing = [f for f in req if f not in d]
print('OK' if not missing else f'MISSING={missing}')
")
if [ "$FIELDS_OK" = "OK" ]; then
    echo "    [PASS] all 5 setting fields present"
else
    echo "    [FAIL] $FIELDS_OK"
    PASS=0
fi

# 辅助：修改某字段 → set_settings → 刷新 CUR_SETTINGS
update_field() {
    local field="$1" value="$2" new_json body
    new_json=$(printf '%s' "$CUR_SETTINGS" | python -c "
import json, sys
d = json.loads(sys.stdin.read())
d['$field'] = json.loads('$value')
print(json.dumps(d))
")
    body=$(printf '{"settings":%s}' "$new_json")
    mcp_call set_settings "$body" | mcp_text > /dev/null 2>&1
    CUR_SETTINGS=$(mcp_call get_settings "{}" | mcp_text)
}

# 辅助：断言当前设置中某字段的值（值比较，避免 0.8 vs 0.80 浮点格式差异）
check_field() {
    local field="$1" value="$2" desc="$3" got
    got=$(printf '%s' "$CUR_SETTINGS" | python -c "
import json, sys
d = json.loads(sys.stdin.read())
print(json.dumps(d['$field']))
" 2>/dev/null)
    if [ "$got" = "$value" ]; then
        echo "    [PASS] $desc (=$value)"
    else
        echo "    [FAIL] $desc: expected $value, got $got"
        PASS=0
    fi
}

# --- 5.2 相似度阈值调整（写回 + 读回验证）---
echo "  5.2 similarity_threshold adjust"
update_field similarity_threshold 0.8
check_field similarity_threshold 0.8 "threshold set to 0.80"
update_field similarity_threshold 0.92
check_field similarity_threshold 0.92 "threshold restored to 0.92"

# --- 5.3 AI 质量评分开关（关闭 + 开启）---
echo "  5.3 AI quality scoring toggle"
update_field ai_enabled false
check_field ai_enabled false "AI scoring disabled"
update_field ai_enabled true
check_field ai_enabled true "AI scoring enabled"

# --- 5.4 增量扫描开关（关闭 + 开启）---
echo "  5.4 incremental scan toggle"
update_field incremental false
check_field incremental false "incremental disabled"
update_field incremental true
check_field incremental true "incremental enabled"

# --- 5.5 增量扫描行为验证：命中缓存二次扫描仍返回完整结果 ---
echo "  5.5 incremental rescan (cache hit)"
RESCAN_RESP=$(mcp_call start_scan "{\"folders\":[\"$SCAN_DIR\"]}")
RESCAN_JSON=$(printf '%s' "$RESCAN_RESP" | mcp_text) || { echo "    [FAIL] incremental rescan: $RESCAN_JSON"; PASS=0; }
RESCAN_TOTAL=$(printf '%s' "$RESCAN_JSON" | python -c "
import json, sys
text = sys.stdin.read()
idx = text.find('{')
if idx < 0:
    raise SystemExit('NO_JSON')
scan = json.loads(text[idx:])
print(scan.get('total_images', 0))
" 2>/dev/null)
RESCAN_TOTAL=${RESCAN_TOTAL:-0}
if [ "$RESCAN_TOTAL" -ge 14 ]; then
    echo "    [PASS] incremental rescan returned $RESCAN_TOTAL images"
else
    echo "    [FAIL] incremental rescan returned $RESCAN_TOTAL (expected >=14)"
    PASS=0
fi

# --- 5.6 恢复默认（防设置污染，影响后续阶段）---
echo "  5.6 restore defaults"
update_field similarity_threshold 0.92
update_field ai_enabled true
update_field incremental true
check_field similarity_threshold 0.92 "threshold default"
check_field ai_enabled true "ai default"
check_field incremental true "incremental default"

# ---------------------------------------------------------------------------
# 阶段 6: 功能链路测试（删除 / 回收站 / 恢复 / 清空）
# ---------------------------------------------------------------------------
echo ""
echo "[6/8] Functional chain test (delete/trash/restore)..."

# --- 5.1 单文件删除（移入临时回收站）---
DEL_FILE="$SCAN_DIR/checker.png"
echo "  5.1 delete single file (to trash): checker.png"
RESP=$(mcp_call delete_files "{\"paths\":[\"$DEL_FILE\"],\"permanent\":false}")
TEXT=$(printf '%s' "$RESP" | mcp_text) || { echo "    [FAIL] delete_files: $TEXT"; PASS=0; }
if [ ! -f "$DEL_FILE" ]; then
    echo "    [PASS] file moved out of scan dir"
else
    echo "    [FAIL] file still exists after delete"
    PASS=0
fi

# --- 5.2 回收站应有 1 条 ---
echo "  5.2 list_trash should have 1 item"
TRASH_JSON=$(mcp_call list_trash "{}")
TRASH_TEXT=$(printf '%s' "$TRASH_JSON" | mcp_text)
TRASH_COUNT=$(printf '%s' "$TRASH_TEXT" | python -c "
import json, sys, re
text = sys.stdin.read()
# text 形如: 临时回收站：N 个文件\n\n[{...}]
m = re.search(r'：(\d+) 个文件', text)
print(m.group(1) if m else '0')
")
if [ "$TRASH_COUNT" -eq 1 ]; then
    echo "    [PASS] trash has 1 item (got $TRASH_COUNT)"
else
    echo "    [FAIL] trash should have 1 item (got $TRASH_COUNT)"
    PASS=0
fi

# 取回收站第一条 id（备用）
TRASH_ID=$(printf '%s' "$TRASH_TEXT" | python -c "
import json, sys, re
text = sys.stdin.read()
# 找到 JSON 数组部分
idx = text.find('[')
if idx >= 0:
    arr = json.loads(text[idx:])
    if arr:
        print(arr[0].get('id', ''))
")

# --- 5.3 恢复单个文件 ---
echo "  5.3 restore_trash_item"
RESTORE_TEXT=$(mcp_call restore_trash_item "{\"id\":\"$TRASH_ID\"}" | mcp_text)
if [ -f "$DEL_FILE" ]; then
    echo "    [PASS] file restored to original path"
else
    echo "    [FAIL] file not restored: $RESTORE_TEXT"
    PASS=0
fi

# --- 5.4 批量删除（3 个）---
echo "  5.4 delete 3 files at once"
BATCH_FILES=("$SCAN_DIR/sunset.jpg" "$SCAN_DIR/landscape.png" "$SCAN_DIR/circles.gif")
BATCH_ARG=$(json_array "${BATCH_FILES[@]}")
BATCH_BODY=$(printf '{"paths":%s,"permanent":false}' "$BATCH_ARG")
RESP=$(mcp_call delete_files "$BATCH_BODY")
TEXT=$(printf '%s' "$RESP" | mcp_text) || { echo "    [FAIL] batch delete: $TEXT"; PASS=0; }
MISSING=0
for f in "${BATCH_FILES[@]}"; do
    [ ! -f "$f" ] && MISSING=$((MISSING + 1))
done
if [ $MISSING -eq 3 ]; then
    echo "    [PASS] all 3 files moved to trash ($TEXT)"
else
    echo "    [FAIL] only $MISSING/3 files moved ($TEXT)"
    PASS=0
fi

# --- 5.5 全部恢复 ---
echo "  5.5 restore_all_trash"
RESTORE_ALL_TEXT=$(mcp_call restore_all_trash "{}" | mcp_text)
BACK=0
for f in "${BATCH_FILES[@]}"; do
    [ -f "$f" ] && BACK=$((BACK + 1))
done
if [ $BACK -eq 3 ]; then
    echo "    [PASS] all 3 files restored ($RESTORE_ALL_TEXT)"
else
    echo "    [FAIL] only $BACK/3 restored ($RESTORE_ALL_TEXT)"
    PASS=0
fi

# --- 5.6 删除后清空回收站（永久删除）---
echo "  5.6 delete then clear_trash (permanent)"
DEL_FILE2="$SCAN_DIR/noise.bmp"
mcp_call delete_files "{\"paths\":[\"$DEL_FILE2\"],\"permanent\":false}" | mcp_text > /dev/null 2>&1
# 清空 + 以最终状态（list_trash==0）断言，避免偶发网络失败导致响应文本不可靠
CLEARED=0
for attempt in 1 2 3; do
    CLEAR_TEXT=$(mcp_call clear_trash "{}" | mcp_text 2>/dev/null)
    TRASH_JSON2=$(mcp_call list_trash "{}")
    TRASH_TEXT2=$(printf '%s' "$TRASH_JSON2" | mcp_text 2>/dev/null)
    TRASH_COUNT2=$(printf '%s' "$TRASH_TEXT2" | python -c "
import re, sys
text = sys.stdin.read()
m = re.search(r'：(\d+) 个文件', text)
print(m.group(1) if m else '0')
" 2>/dev/null)
    TRASH_COUNT2=${TRASH_COUNT2:-0}
    if [ "$TRASH_COUNT2" -eq 0 ]; then CLEARED=1; break; fi
    sleep 1
done
if [ $CLEARED -eq 1 ] && [ ! -f "$DEL_FILE2" ]; then
    echo "    [PASS] trash cleared, file permanently deleted ($CLEAR_TEXT)"
else
    echo "    [FAIL] clear_trash: $CLEAR_TEXT (trash count=$TRASH_COUNT2)"
    PASS=0
fi

# 验证回收站为空（若上面已断言 count==0 则冗余但无害）
if [ "$TRASH_COUNT2" -eq 0 ]; then
    echo "    [PASS] trash is empty after clear"
else
    echo "    [FAIL] trash should be empty (got $TRASH_COUNT2)"
    PASS=0
fi

# 恢复被永久删除的文件以便后续导出？（clear_trash 是永久的，noise.bmp 已丢失 → 重新复制）
cp "$SAMPLE_DIR/noise.bmp" "$SCAN_DIR/" 2>/dev/null

# ---------------------------------------------------------------------------
# 阶段 6: 导出报告 + 清理
# ---------------------------------------------------------------------------
echo ""
echo "[7/8] Export report + cleanup..."
EXPORT_BODY=$(printf '{"output_path":"%s"}' "$REPORT")
REPORTED=0
for attempt in 1 2 3; do
    mcp_call export_report "$EXPORT_BODY" | mcp_text > /dev/null 2>&1
    if [ -f "$REPORT" ]; then REPORTED=1; break; fi
    sleep 1
done
if [ $REPORTED -eq 1 ]; then
    echo "  [PASS] Report: $REPORT"
else
    echo "  [WARN] Report not exported"
fi

# 清理 cache
mcp_call clear_cache "{}" | mcp_text > /dev/null 2>&1

# 停应用（双重确认，nohup 启动的进程可能延迟回收）
kill_pixsweep; sleep 1
sleep 1
kill_pixsweep; sleep 1
sleep 1

# 清理临时目录
rm -rf "$TEMP_ROOT"
echo "  Temp dir removed: $TEMP_ROOT"

# trap 确保任何异常路径都关掉 PixSweep + 清理临时目录
cleanup_on_exit() {
    kill_pixsweep; sleep 1
    rm -rf "$TEMP_ROOT" 2>/dev/null || true
}
trap cleanup_on_exit EXIT

# ---------------------------------------------------------------------------
# 阶段 7: 报告
# ---------------------------------------------------------------------------
echo ""
echo "[8/8] Result"
if [ $PASS -eq 1 ]; then
    echo "====== TEST PASSED ======"
    exit 0
else
    echo "====== TEST FAILED ======"
    exit 1
fi
