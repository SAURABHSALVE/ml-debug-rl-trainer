function ScoreBar({ label, value, max = 0.5, color }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="score-bar-row">
      <span className="score-bar-label">{label}</span>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="score-bar-val">{value.toFixed(3)}</span>
    </div>
  );
}

function scoreColor(s) {
  if (s >= 0.8) return "#4ade80";
  if (s >= 0.5) return "#facc15";
  return "#f87171";
}

export default function RewardPanel({ reward, info }) {
  const {
    score,
    fix_quality,
    efficiency_bonus,
    intermediate_signal,
    cumulative_episode_signal,
    reasoning
  } = reward;

  const gb = info?.grader_breakdown ?? {};
  const diagnosis_score = gb.diagnosis_score ?? 0;
  const fix_score = gb.fix_score ?? 0;
  const color = scoreColor(score);

  // Check if this is an investigation action (non-terminal)
  const isInvestigation = intermediate_signal !== undefined && intermediate_signal !== null;

  if (isInvestigation) {
    return (
      <div className="reward-panel">
        <div className="reward-header">
          <h3 className="section-title">Action Signal</h3>
          <div className="total-score" style={{
            color: intermediate_signal > 0 ? "#4ade80" : intermediate_signal < 0 ? "#f87171" : "#6b7280"
          }}>
            {intermediate_signal >= 0 ? "+" : ""}{intermediate_signal.toFixed(3)}
          </div>
        </div>

        <div className="signal-breakdown">
          <div className="signal-row">
            <span>This action:</span>
            <span className={intermediate_signal > 0 ? "signal-positive" : intermediate_signal < 0 ? "signal-negative" : "signal-neutral"}>
              {intermediate_signal > 0 ? "+" : ""}{intermediate_signal.toFixed(3)}
            </span>
          </div>
          <div className="signal-row">
            <span>Episode total:</span>
            <span>{cumulative_episode_signal.toFixed(3)}</span>
          </div>
        </div>

        <div className="reasoning-box">
          <div className="reason-line">{reasoning}</div>
        </div>
      </div>
    );
  }

  // Terminal action (diagnose) - original logic
  const bonusApplied = reasoning?.includes("trajectory bonus");
  const reasons = reasoning?.split(" | ").filter(Boolean) || [];

  return (
    <div className="reward-panel">
      <div className="reward-header">
        <h3 className="section-title">Diagnosis Result</h3>
        <div className="total-score" style={{ color }}>
          {score.toFixed(3)}
          <span className="score-max"> / 1.0</span>
        </div>
      </div>

      <ScoreBar label="Diagnosis" value={diagnosis_score} max={0.5} color="#60a5fa" />
      <ScoreBar label="Fix Quality" value={fix_score} max={0.5} color="#a78bfa" />
      <ScoreBar label="Fix Score" value={fix_quality ?? 0} max={1.0} color={color} />
      {efficiency_bonus > 0 && (
        <ScoreBar label="Efficiency Bonus" value={efficiency_bonus} max={0.05} color="#34d399" />
      )}
      <ScoreBar label="Total" value={score} max={1.0} color={color} />

      {bonusApplied && (
        <div className="bonus-tag">⚡ Trajectory bonus applied (+0.05)</div>
      )}

      <div className="reasoning-box">
        <h4 className="reasoning-title">Grader Reasoning</h4>
        {reasons.map((r, i) => (
          <div
            key={i}
            className={`reason-line ${
              r.startsWith("✓") ? "reason-good" : r.startsWith("✗") ? "reason-bad" : "reason-warn"
            }`}
          >
            {r}
          </div>
        ))}
      </div>

      {info?.ground_truth_bug_type && (
        <div className="ground-truth">
          <span className="gt-label">Bug type: </span>
          <code>{info.ground_truth_bug_type}</code>
        </div>
      )}
    </div>
  );
}
