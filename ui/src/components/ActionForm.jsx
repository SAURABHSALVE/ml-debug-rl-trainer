const ACTION_TYPES = [
  { value: "fetch_logs", label: "Fetch Training Logs", terminal: false },
  { value: "fetch_config", label: "Fetch Config Values", terminal: false },
  { value: "fetch_loss_curve", label: "Fetch Loss Curve", terminal: false },
  { value: "fetch_diagnostics", label: "Run Diagnostics", terminal: false },
  { value: "fetch_class_data", label: "Fetch Class Data", terminal: false },
  { value: "diagnose", label: "Submit Diagnosis", terminal: true },
];

const FIX_TYPES = [
  { value: "config_change", label: "Config Change" },
  { value: "data_fix", label: "Data Fix" },
  { value: "architecture_change", label: "Architecture Change" },
];

const DIAGNOSTIC_CHECKS = [
  { value: "overfitting", label: "Overfitting Analysis" },
  { value: "gradients", label: "Gradient Analysis" },
  { value: "trends", label: "Training Trends" },
  { value: "class_balance", label: "Class Balance" },
];

const HINTS = {
  easy: {
    diagnosis: 'e.g. "The model is overfitting — val loss diverges after epoch 10 while train loss keeps dropping."',
    fix_detail: 'e.g. "Add dropout=0.3 and weight_decay=1e-4 to prevent overfitting."',
  },
  medium: {
    diagnosis: 'e.g. "Learning rate is too high (0.5) causing gradient explosion and NaN losses by epoch 11."',
    fix_detail: 'e.g. "Reduce lr from 0.5 to 0.001 and add gradient_clip=1.0 and warmup_steps=500."',
  },
  hard: {
    diagnosis: 'e.g. "Silent label corruption in class_2 — per-class val accuracy stagnates at ~0.35 while others reach 0.90+."',
    fix_detail: 'e.g. "Re-annotate class_2 training samples. Run label consistency audit on batch 47 onwards."',
  },
};

export default function ActionForm({ action, onChange, onSubmit, loading, difficulty }) {
  const hints = HINTS[difficulty] || HINTS.easy;

  const set = (key) => (e) =>
    onChange((a) => ({ ...a, [key]: e.target.value }));

  const setActionType = (actionType) => {
    // Reset action parameters when changing type
    const newAction = { action_type: actionType };
    if (actionType === "diagnose") {
      newAction.diagnosis = "";
      newAction.fix_type = "config_change";
      newAction.fix_detail = "";
      newAction.confidence = 0.8;
    }
    onChange(newAction);
  };

  const isTerminal = ACTION_TYPES.find(t => t.value === action.action_type)?.terminal;

  return (
    <div className="action-form">
      <h3 className="section-title">Choose Action</h3>

      {/* Action Type Selector */}
      <label className="field-label">Action Type</label>
      <select
        className="field-select"
        value={action.action_type}
        onChange={(e) => setActionType(e.target.value)}
      >
        {ACTION_TYPES.map((type) => (
          <option key={type.value} value={type.value}>
            {type.label}
          </option>
        ))}
      </select>

      {/* Parameters based on action type */}
      {action.action_type === "fetch_logs" && (
        <div className="field-group">
          <label className="field-label">Epochs</label>
          <input
            type="text"
            className="field-input"
            placeholder="e.g. 15-20, all, 1-10"
            value={action.epochs || ""}
            onChange={set("epochs")}
          />
        </div>
      )}

      {action.action_type === "fetch_config" && (
        <div className="field-group">
          <label className="field-label">Config Keys (comma-separated)</label>
          <input
            type="text"
            className="field-input"
            placeholder="e.g. lr, optimizer, batch_size"
            value={action.keys ? action.keys.join(", ") : ""}
            onChange={(e) => onChange((a) => ({
              ...a,
              keys: e.target.value ? e.target.value.split(",").map(k => k.trim()) : null
            }))}
          />
        </div>
      )}

      {action.action_type === "fetch_loss_curve" && (
        <div className="field-group">
          <label className="field-label">Split</label>
          <select
            className="field-select"
            value={action.split || ""}
            onChange={set("split")}
          >
            <option value="">Select split...</option>
            <option value="train">Train</option>
            <option value="val">Validation</option>
            <option value="all">All</option>
          </select>
        </div>
      )}

      {action.action_type === "fetch_diagnostics" && (
        <div className="field-group">
          <label className="field-label">Check Type</label>
          <select
            className="field-select"
            value={action.check || ""}
            onChange={set("check")}
          >
            <option value="">Select check...</option>
            {DIAGNOSTIC_CHECKS.map((check) => (
              <option key={check.value} value={check.value}>
                {check.label}
              </option>
            ))}
          </select>
        </div>
      )}

      {action.action_type === "fetch_class_data" && (
        <div className="field-group">
          <label className="field-label">Class ID</label>
          <input
            type="number"
            className="field-input"
            placeholder="e.g. 2"
            min="0"
            max="4"
            value={action.class_id || ""}
            onChange={(e) => onChange((a) => ({
              ...a,
              class_id: e.target.value ? parseInt(e.target.value) : null
            }))}
          />
        </div>
      )}

      {/* Diagnosis fields - only for diagnose action */}
      {action.action_type === "diagnose" && (
        <>
          <label className="field-label">
            Root Cause Diagnosis
            <span className="required">*</span>
          </label>
          <textarea
            className="field-textarea"
            rows={3}
            placeholder={hints.diagnosis}
            value={action.diagnosis}
            onChange={set("diagnosis")}
          />

          <div className="field-row">
            <div className="field-group">
              <label className="field-label">Fix Type</label>
              <select
                className="field-select"
                value={action.fix_type}
                onChange={set("fix_type")}
              >
                {FIX_TYPES.map((f) => (
                  <option key={f.value} value={f.value}>
                    {f.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="field-group confidence-group">
              <label className="field-label">
                Confidence &nbsp;
                <strong>{(+action.confidence).toFixed(2)}</strong>
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={action.confidence}
                onChange={set("confidence")}
                className="confidence-slider"
              />
            </div>
          </div>

          <label className="field-label">
            Fix Detail (include specific values)
            <span className="required">*</span>
          </label>
          <textarea
            className="field-textarea"
            rows={3}
            placeholder={hints.fix_detail}
            value={action.fix_detail}
            onChange={set("fix_detail")}
          />
        </>
      )}

      <button
        className="btn btn-submit"
        onClick={onSubmit}
        disabled={loading || (action.action_type === "diagnose" && (!action.diagnosis.trim() || !action.fix_detail.trim()))}
      >
        {loading ? "Executing…" : isTerminal ? "Submit Diagnosis →" : "Execute Action →"}
      </button>
    </div>
  );
}
