function scoreColor(s) {
  if (s >= 0.8) return "#4ade80";
  if (s >= 0.5) return "#facc15";
  return "#f87171";
}

function ScoreCircle({ score }) {
  const color = scoreColor(score);
  const r = 40;
  const circ = 2 * Math.PI * r;
  const offset = circ - score * circ;
  return (
    <svg width={100} height={100} viewBox="0 0 100 100">
      <circle cx={50} cy={50} r={r} fill="none" stroke="#1e293b" strokeWidth={10} />
      <circle
        cx={50}
        cy={50}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={10}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform="rotate(-90 50 50)"
        style={{ transition: "stroke-dashoffset 0.8s ease" }}
      />
      <text x={50} y={54} textAnchor="middle" fill={color} fontSize={18} fontWeight="bold">
        {score.toFixed(2)}
      </text>
    </svg>
  );
}

const DIFFICULTY_LABELS = { easy: "Overfitting", medium: "LR Explosion", hard: "Data Poisoning" };

export default function EpisodeSummary({ summary, history, onRestart }) {
  const { easy_score, medium_score, hard_score, average_score, trajectory_bonus_applied } = summary;
  const scores = [easy_score, medium_score, hard_score];
  const labels = ["Easy", "Medium", "Hard"];

  return (
    <div className="summary-overlay">
      <div className="summary-card">
        <h2 className="summary-title">Episode Complete 🎉</h2>

        <div className="summary-circles">
          {scores.map((s, i) => (
            <div key={i} className="circle-block">
              <ScoreCircle score={s} />
              <div className={`badge ${labels[i].toLowerCase()}`}>{labels[i]}</div>
              <div className="circle-task">{DIFFICULTY_LABELS[labels[i].toLowerCase()]}</div>
            </div>
          ))}
        </div>

        <div className="avg-row">
          <span className="avg-label">Average Score</span>
          <span className="avg-val" style={{ color: scoreColor(average_score) }}>
            {average_score.toFixed(3)}
          </span>
        </div>

        {trajectory_bonus_applied && (
          <div className="bonus-tag">⚡ Trajectory bonus was applied on the hard task</div>
        )}

        <div className="history-table">
          <h3>Per-Task Breakdown</h3>
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Difficulty</th>
                <th>Diagnosis</th>
                <th>Fix</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td><code>{h.task_id}</code></td>
                  <td><span className={`badge ${h.difficulty}`}>{h.difficulty}</span></td>
                  <td>{h.diagnosis_score.toFixed(3)}</td>
                  <td>{h.fix_score.toFixed(3)}</td>
                  <td style={{ color: scoreColor(h.score), fontWeight: 700 }}>
                    {h.score.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="summary-rating">
          {average_score >= 0.8
            ? "🏆 Expert Debugger! Excellent diagnosis across all tasks."
            : average_score >= 0.6
            ? "👍 Good job! A few signals were missed — check the reasoning panels."
            : average_score >= 0.4
            ? "⚠ Partial credit. Review the logs more carefully for hidden signals."
            : "🔴 Needs work. Try reading all log lines and config values closely."}
        </div>

        <button className="btn btn-start" onClick={onRestart}>
          ↺ Try Again
        </button>
      </div>
    </div>
  );
}
