import { useState, useCallback } from "react";
import TaskPanel from "./components/TaskPanel";
import ActionForm from "./components/ActionForm";
import RewardPanel from "./components/RewardPanel";
import EpisodeSummary from "./components/EpisodeSummary";
import AIAdvisor from "./components/AIAdvisor";

const API = "/api";

const INITIAL_ACTION = {
  action_type: "fetch_logs",
  epochs: "1-10",
};

export default function App() {
  const [obs, setObs] = useState(null);
  const [reward, setReward] = useState(null);
  const [info, setInfo] = useState(null);
  const [done, setDone] = useState(false);
  const [action, setAction] = useState(INITIAL_ACTION);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);   // [{task_id, score, reasoning}]
  const [episodeSummary, setEpisodeSummary] = useState(null);
  const [started, setStarted] = useState(false);

  const taskIndex = history.length + 1;

  const handleReset = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/reset`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setObs(data);
      setReward(null);
      setInfo(null);
      setDone(false);
      setAction(INITIAL_ACTION);
      setHistory([]);
      setEpisodeSummary(null);
      setStarted(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleStep = useCallback(async () => {
    // Validation for diagnose action
    if (action.action_type === "diagnose" && (!action.diagnosis.trim() || !action.fix_detail.trim())) {
      setError("Diagnosis and Fix Detail are required.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(action),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      const data = await res.json();

      // Handle terminal action (diagnose)
      if (action.action_type === "diagnose") {
        const gb = data.info?.grader_breakdown ?? {};
        setHistory((h) => [
          ...h,
          {
            task_id: data.info.task_id,
            difficulty: data.info.difficulty,
            score: data.reward.score,
            fix_quality: data.reward.fix_quality,
            efficiency_bonus: data.reward.efficiency_bonus,
            diagnosis_score: gb.diagnosis_score ?? 0,
            fix_score: gb.fix_score ?? 0,
            reasoning: data.reward.reasoning,
          },
        ]);

        if (data.done && data.info.episode_summary) {
          setEpisodeSummary(data.info.episode_summary);
        }
      }

      setObs(data.observation);
      setReward(data.reward);
      setInfo(data.info);
      setDone(data.done);

      // Reset action only for terminal actions
      if (action.action_type === "diagnose") {
        setAction(INITIAL_ACTION);
      }

    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [action]);

  const handleActionSelected = useCallback((actionName) => {
    // Populate the ActionForm with the suggested action
    const defaultParams = {
      fetch_logs: { action_type: "fetch_logs", epochs: "1-10" },
      fetch_config: { action_type: "fetch_config", keys: ["lr", "optimizer", "batch_size"] },
      fetch_loss_curve: { action_type: "fetch_loss_curve", split: "all" },
      fetch_diagnostics: { action_type: "fetch_diagnostics", check: "overfitting" },
      fetch_class_data: { action_type: "fetch_class_data", class_id: 0 },
    };
    
    if (defaultParams[actionName]) {
      setAction(defaultParams[actionName]);
    }
  }, []);

  const difficultyOf = obs?.difficulty;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="logo">🔬</span>
          <div>
            <h1>ML Experiment Debugger</h1>
            <p className="subtitle">OpenEnv · Team AIverse</p>
          </div>
        </div>
        <div className="header-right">
          {started && !done && (
            <div className="progress-steps">
              {[1, 2, 3].map((n) => (
                <div
                  key={n}
                  className={`step-dot ${
                    history.length >= n
                      ? "step-done"
                      : taskIndex === n
                      ? "step-active"
                      : ""
                  }`}
                >
                  {n}
                </div>
              ))}
            </div>
          )}
          <button
            className="btn btn-reset"
            onClick={handleReset}
            disabled={loading}
          >
            {loading && !started ? "Starting…" : started ? "↺ Restart" : "▶ Start Episode"}
          </button>
        </div>
      </header>

      {/* ── Error banner ── */}
      {error && (
        <div className="error-banner">
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* ── Welcome screen ── */}
      {!started && (
        <div className="welcome">
          <div className="welcome-card">
            <h2>Welcome to ML Experiment Debugger</h2>
            <p>
              Act as an expert ML engineer. You'll receive training logs, loss
              curves, config YAMLs, and GPU metrics from broken model runs.
              Diagnose what went wrong and prescribe the fix.
            </p>
            <div className="task-previews">
              <div className="preview-card easy">
                <span className="badge easy">Easy</span>
                <strong>Overfitting Detection</strong>
                <p>Train loss drops, val loss diverges. Classic regularization bug.</p>
              </div>
              <div className="preview-card medium">
                <span className="badge medium">Medium</span>
                <strong>LR Explosion</strong>
                <p>Loss goes NaN mid-training. Spot the misconfigured hyperparameter.</p>
              </div>
              <div className="preview-card hard">
                <span className="badge hard">Hard</span>
                <strong>Silent Data Poisoning</strong>
                <p>Global metrics look fine. One class is secretly corrupted.</p>
              </div>
            </div>
            <button className="btn btn-start" onClick={handleReset} disabled={loading}>
              {loading ? "Starting…" : "▶ Start Episode"}
            </button>
          </div>
        </div>
      )}

      {/* ── Episode done ── */}
      {done && episodeSummary && (
        <EpisodeSummary
          summary={episodeSummary}
          history={history}
          onRestart={handleReset}
        />
      )}

      {/* ── Active task ── */}
      {started && !done && obs && obs.task_id !== "done" && (
        <div className="main-grid">
          {/* Left column */}
          <div className="left-col">
            <div className="task-header">
              <span className={`badge ${difficultyOf}`}>
                {difficultyOf?.toUpperCase()}
              </span>
              <span className="task-num">Task {taskIndex} / 3</span>
              <span className="task-id">{obs.task_id}</span>
            </div>
            <TaskPanel obs={obs} />
          </div>

          {/* Right column */}
          <div className="right-col">
            {/* AI Advisor */}
            <AIAdvisor 
              taskDifficulty={difficultyOf}
              stepsUsed={reward?.steps_used || 0}
              stepsRemaining={15 - (reward?.steps_used || 0)}
              onActionSelected={handleActionSelected}
            />

            {/* Scores so far */}
            {history.length > 0 && (
              <div className="history-scores">
                <h3>Scores So Far</h3>
                {history.map((h, i) => (
                  <div key={i} className="history-row">
                    <span className={`badge ${h.difficulty}`}>{h.difficulty}</span>
                    <span className="history-task">{h.task_id}</span>
                    <span className="history-score">{h.score.toFixed(3)}</span>
                    <div className="mini-bar">
                      <div
                        className="mini-fill"
                        style={{ width: `${h.score * 100}%`, background: scoreColor(h.score) }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            <ActionForm
              action={action}
              onChange={setAction}
              onSubmit={handleStep}
              loading={loading}
              difficulty={difficultyOf}
            />

            {reward && (
              <RewardPanel reward={reward} info={info} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function scoreColor(s) {
  if (s >= 0.8) return "#4ade80";
  if (s >= 0.5) return "#facc15";
  return "#f87171";
}
