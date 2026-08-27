/**
 * 评分标准说明面板
 *
 * 介绍 PixSweep 的评分体系：先判类型，再按类型分路评分，最后成组推荐最佳。
 * 帮助用户理解"为什么这张推荐了 / 为什么标失焦 / 为什么标闭眼"。
 */

interface ScoreHelpModalProps {
  onClose: () => void;
}

export function ScoreHelpModal({ onClose }: ScoreHelpModalProps) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal-wide modal-fixed-title score-help-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-titlebar">
          <h2 className="modal-title">评分标准说明</h2>
          <button className="modal-close-btn" onClick={onClose} aria-label="关闭" title="关闭（Esc）">
            ✕
          </button>
        </div>

        <div className="modal-body score-help-body">
          <div className="score-help-intro">
            PixSweep 先给每组图片判<b>类型</b>，再按类型分路评分，最终用<b>综合评分</b>选出
            组内"最好的一张"推荐保留。卡片上显示的标签：<b>类型 / 闭眼 / 失焦 / 综合</b>。
          </div>

          <div className="score-help-section">
            <div className="score-help-section-head type">
              <span className="score-help-section-name">类型</span>
              <span className="score-help-section-range">人像 · 风景 · 宠物 · 其他</span>
            </div>
            <div className="score-help-section-body">
              <div className="score-help-row">
                <b>来源</b>：MobileNetV3 场景分类 + 人脸检测（检测到人脸 → 判为「人像」）
              </div>
              <div className="score-help-row">
                <b>作用</b>：决定走哪条评分路径（人像走眼睛/人脸专评，其余走对焦 + 美学）
              </div>
            </div>
          </div>

          <div className="score-help-section">
            <div className="score-help-section-head focus">
              <span className="score-help-section-name">对焦（是否失焦）</span>
              <span className="score-help-section-range">1.0 ~ 10.0</span>
            </div>
            <div className="score-help-section-body">
              <div className="score-help-row">
                <b>来源</b>：灰度图拉普拉斯方差（锐度/清晰度，模型无关）
              </div>
              <div className="score-help-row">
                <b>范围</b>：人像/宠物取<b>眼部</b>对焦；风景/其他取<b>整图</b>对焦
              </div>
              <div className="score-help-row warn">
                <b>失焦</b>：对焦分低于阈值 → 标「失焦」（只标明显低清晰度）
              </div>
            </div>
          </div>

          <div className="score-help-section">
            <div className="score-help-section-head eye">
              <span className="score-help-section-name">闭眼</span>
              <span className="score-help-section-range">对人像/宠物</span>
            </div>
            <div className="score-help-section-body">
              <div className="score-help-row">
                <b>来源</b>：OCEC 闭眼检测（在检测到的眼睛 ROI 上判定）
              </div>
              <div className="score-help-row">
                <b>规则</b>：<b>双眼都判闭</b>才标「闭眼」并降权（单眼噪声不会误伤睁眼）
              </div>
            </div>
          </div>

          <div className="score-help-section">
            <div className="score-help-section-head composite">
              <span className="score-help-section-name">综合评分</span>
              <span className="score-help-section-range">1.0 ~ 10.0</span>
            </div>
            <div className="score-help-section-body">
              <div className="score-help-row">
                <b>人像</b>：人像美学(人脸分) 0.55 主导 + 眼部对焦 0.30 + 启发式 0.15，再×闭眼降权
              </div>
              <div className="score-help-row">
                <b>风景 / 宠物</b>：美学 + 对焦均衡（对焦略高/均衡）+ 启发式
              </div>
              <div className="score-help-row">
                <b>其他</b>：美学 0.25 + 对焦 0.60 + 启发式 0.15（对焦最主导）
              </div>
              <div className="score-help-row">
                <b>启发式</b>：分辨率/文件大小越高分越高（同内容时大文件通常更完整）
              </div>
              <div className="score-help-row warn">
                <b>目的</b>：组内内容相同，评分用来区分"哪张拍得更好"，推荐综合分最高者
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
