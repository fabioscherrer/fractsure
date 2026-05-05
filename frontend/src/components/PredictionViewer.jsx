import { useEffect, useMemo, useState } from "react";
import { AlertCircle, ImagePlus, LoaderCircle } from "lucide-react";

function clampBox(box) {
  const x1 = Number(box.x1) || 0;
  const y1 = Number(box.y1) || 0;
  const x2 = Number(box.x2) || 0;
  const y2 = Number(box.y2) || 0;

  return {
    ...box,
    x1: Math.min(x1, x2),
    y1: Math.min(y1, y2),
    x2: Math.max(x1, x2),
    y2: Math.max(y1, y2),
  };
}

export function PredictionViewer({
  error,
  imageName,
  imageUrl,
  isLoading,
  prediction,
  activeFindingIndex,
  onActiveFindingChange,
}) {
  const [imageSize, setImageSize] = useState(null);
  const boxes = useMemo(() => (prediction?.boxes || []).map(clampBox), [prediction]);

  useEffect(() => {
    setImageSize(null);
  }, [imageUrl]);

  if (!imageUrl) {
    return (
      <section className="viewer-panel" aria-label="Image preview">
        <div className="viewer-toolbar">
          <div className="viewer-title">
            <span className="step-label">Step 2</span>
            <h2>Preview & Findings</h2>
          </div>
        </div>

        <div className="empty-viewer">
          <ImagePlus size={42} />
          <p>No image selected</p>
        </div>
      </section>
    );
  }

  const strokeWidth = imageSize ? Math.max(4, Math.round(imageSize.width * 0.006)) : 5;
  const labelFontSize = imageSize ? Math.max(22, Math.round(imageSize.width * 0.02)) : 22;
  const labelHeight = Math.round(labelFontSize * 1.45);
  const labelGap = Math.round(labelFontSize * 0.35);

  return (
    <section className="viewer-panel" aria-label="Image preview">
      <div className="viewer-toolbar">
        <div className="viewer-title">
          <span className="step-label">Step 2</span>
          <h2>Preview & Findings</h2>
        </div>
        <span className="image-name" title={imageName}>{imageName}</span>
        {prediction && <strong>{boxes.length} Findings</strong>}
      </div>

      <div className="image-stage">
        <div className="image-frame">
          <img
            className="preview-image"
            src={imageUrl}
            alt={imageName || "Selected X-ray image"}
            onLoad={(event) => {
              setImageSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              });
            }}
          />

          {imageSize && boxes.length > 0 && (
            <svg
              className="prediction-overlay"
              viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
              preserveAspectRatio="none"
              aria-label="Detected findings"
            >
              {boxes.map((box, index) => {
                const label = `${box.label || "fracture"} ${Number(box.score || 0).toFixed(2)}`;
                const isActive = activeFindingIndex === index;
                const rawLabelWidth = Math.max(labelFontSize * 6.5, label.length * labelFontSize * 0.64);
                const labelWidth = Math.min(imageSize.width, rawLabelWidth);
                const labelX = Math.min(box.x1, Math.max(0, imageSize.width - labelWidth));
                const hasRoomAbove = box.y1 >= labelHeight + labelGap;
                const belowBoxY = box.y2 + labelGap;
                const hasRoomBelow = belowBoxY + labelHeight <= imageSize.height;
                const labelY = hasRoomAbove
                  ? box.y1 - labelHeight - labelGap
                  : hasRoomBelow
                    ? belowBoxY
                    : Math.min(Math.max(0, box.y1 + labelGap), imageSize.height - labelHeight);

                return (
                  <g
                    key={`${box.x1}-${box.y1}-${index}`}
                    className={`detection-overlay${isActive ? " is-active" : ""}`}
                    tabIndex="0"
                    onMouseEnter={() => onActiveFindingChange(index)}
                    onMouseLeave={() => onActiveFindingChange(null)}
                    onFocus={() => onActiveFindingChange(index)}
                    onBlur={() => onActiveFindingChange(null)}
                  >
                    <rect
                      className="box-shape"
                      x={box.x1}
                      y={box.y1}
                      width={Math.max(1, box.x2 - box.x1)}
                      height={Math.max(1, box.y2 - box.y1)}
                      strokeWidth={strokeWidth}
                    />
                    <rect
                      className="box-label-bg"
                      x={labelX}
                      y={labelY}
                      width={labelWidth}
                      height={labelHeight}
                      rx="4"
                    />
                    <text
                      className="box-label"
                      x={labelX + labelFontSize * 0.5}
                      y={labelY + labelHeight * 0.72}
                      fontSize={labelFontSize}
                    >
                      {label}
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        {isLoading && (
          <div className="image-state">
            <LoaderCircle className="spin" size={30} />
            <span>Analysis running</span>
          </div>
        )}
      </div>

      {error && (
        <div className="inline-error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}
    </section>
  );
}
