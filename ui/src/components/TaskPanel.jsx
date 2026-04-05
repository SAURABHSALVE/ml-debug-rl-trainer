import LossCurveChart from "./LossCurveChart";

export default function TaskPanel({ obs }) {
  const {
    description,
    training_logs,
    config_yaml,
    loss_curve,
    gpu_metrics,
    action_result,
    action_history,
    steps_remaining,
    max_steps,
    hint,
  } = obs;

  const mainCurves = loss_curve
    ? { train: loss_curve.train, val: loss_curve.val }
    : null;
  const classCurves = loss_curve
    ? Object.entries(loss_curve).filter(([k]) => k !== "train" && k !== "val")
    : [];

  return (
    <div className="task-panel">
      {/* Description */}
      <section className="panel-section">
        <h3 className="section-title">Task Description</h3>
        <p className="task-description">{description}</p>
        <div className="step-budget">
          Steps remaining: <strong>{steps_remaining}</strong> / {max_steps}
        </div>
      </section>

      {/* Last tool result */}
      {action_result && (
        <section className="panel-section">
          <h3 className="section-title">🔍 Action Result</h3>
          <pre className="config-box action-result">{action_result}</pre>
        </section>
      )}

      {/* Action history */}
      {action_history?.length > 0 && (
        <section className="panel-section">
          <h3 className="section-title">Action History</h3>
          <div className="log-box" style={{ maxHeight: "120px" }}>
            {action_history.map((a, i) => (
              <div key={i} className="log-line action-history-item">{a}</div>
            ))}
          </div>
        </section>
      )}

      {/* Hint */}
      {hint && (
        <section className="panel-section">
          <div className="hint-box">💡 {hint}</div>
        </section>
      )}

      {/* Loss Curve — shown on task entry */}
      {mainCurves && (
        <section className="panel-section">
          <h3 className="section-title">Loss Curves</h3>
          <LossCurveChart curves={mainCurves} label="Train / Val Loss" />
          {classCurves.length > 0 && (
            <>
              <h4 className="section-subtitle">Per-Class Accuracy</h4>
              <LossCurveChart
                curves={Object.fromEntries(classCurves)}
                label="Per-Class Val Accuracy"
                yLabel="Accuracy"
              />
            </>
          )}
        </section>
      )}

      {/* GPU Metrics */}
      {gpu_metrics && (
        <section className="panel-section">
          <h3 className="section-title">GPU Metrics</h3>
          <div className="gpu-grid">
            {Object.entries(gpu_metrics).map(([key, vals]) => {
              const last = vals[vals.length - 1]?.toFixed(1);
              const avg = (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1);
              return (
                <div key={key} className="gpu-card">
                  <span className="gpu-label">{key.replace(/_/g, " ")}</span>
                  <span className="gpu-val">{last}</span>
                  <span className="gpu-avg">avg {avg}</span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Training Logs */}
      {training_logs && (
        <section className="panel-section">
          <h3 className="section-title">
            Training Logs
            <span className="log-count">{training_logs.length} epochs</span>
          </h3>
          <div className="log-box">
            {training_logs.map((line, i) => (
              <div
                key={i}
                className={`log-line ${
                  line.includes("nan") || line.includes("UNSTABLE") || line.includes("WARNING")
                    ? "log-warn"
                    : ""
                }`}
              >
                {line}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Config YAML */}
      {config_yaml && (
        <section className="panel-section">
          <h3 className="section-title">Config YAML</h3>
          <pre className="config-box">{config_yaml}</pre>
        </section>
      )}
    </div>
  );
}
