import { CheckCircle2, CircleDashed, Layers3, RotateCcw } from "lucide-react";

export function ResultPanel({ prediction, activeFindingIndex, onActiveFindingChange, onResetDetection }) {
  const boxes = prediction?.boxes || [];

  return (
    <aside className="result-panel" aria-label="Result details">
      <div className="panel-heading">
        <Layers3 size={18} />
        <div>
          <span className="step-label">Step 3</span>
          <h2>Results</h2>
        </div>
      </div>

      {!prediction ? (
        <div className="result-empty">
          <CircleDashed size={28} />
          <p>Ready for the next analysis</p>
        </div>
      ) : (
        <>
          <div className="metric-grid">
            <div className="metric">
              <span>Findings</span>
              <strong>{boxes.length}</strong>
            </div>
            <div className="metric">
              <span>Model</span>
              <strong>{prediction.model_loaded ? "ONNX" : "Demo"}</strong>
            </div>
          </div>

          <div className="model-note">
            <CheckCircle2 size={17} />
            <span>{prediction.model_loaded ? "Model response received" : "Placeholder output active"}</span>
          </div>

          <div className="detection-list">
            {boxes.length === 0 ? (
              <p className="muted-copy">No boxes returned.</p>
            ) : (
              boxes.map((box, index) => (
                <div
                  className={`detection-row${activeFindingIndex === index ? " is-active" : ""}`}
                  key={`${box.x1}-${box.y1}-${index}`}
                  tabIndex="0"
                  onMouseEnter={() => onActiveFindingChange(index)}
                  onMouseLeave={() => onActiveFindingChange(null)}
                  onFocus={() => onActiveFindingChange(index)}
                  onBlur={() => onActiveFindingChange(null)}
                >
                  <div>
                    <strong>{box.label || "fracture"}</strong>
                    <span>
                      {Math.round(box.x1)}, {Math.round(box.y1)} - {Math.round(box.x2)}, {Math.round(box.y2)}
                    </span>
                  </div>
                  <b>{Number(box.score || 0).toFixed(2)}</b>
                </div>
              ))
            )}
          </div>

          <button className="reset-action" type="button" onClick={onResetDetection}>
            <RotateCcw size={17} />
            <span>Start new detection</span>
          </button>
        </>
      )}
    </aside>
  );
}
