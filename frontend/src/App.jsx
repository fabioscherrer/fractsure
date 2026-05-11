import { useEffect, useState } from "react";
import { Info, ScanSearch, ShieldCheck } from "lucide-react";

import fractsureLogo from "./assets/fractsure_logo.png";
import { predictFracture } from "./api/fractureApi";
import { FileDropzone } from "./components/FileDropzone";
import { PredictionViewer } from "./components/PredictionViewer";
import { ResultPanel } from "./components/ResultPanel";
import "./styles/app.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [activeFindingIndex, setActiveFindingIndex] = useState(null);

  useEffect(() => {
    if (!selectedFile) {
      setImageUrl("");
      return undefined;
    }

    const nextImageUrl = URL.createObjectURL(selectedFile);
    setImageUrl(nextImageUrl);

    return () => {
      URL.revokeObjectURL(nextImageUrl);
    };
  }, [selectedFile]);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setPrediction(null);
    setActiveFindingIndex(null);
    setError("");
  };

  const handleRunDetection = async () => {
    if (!selectedFile || isPredicting) {
      return;
    }

    setIsPredicting(true);
    setError("");

    try {
      const result = await predictFracture(selectedFile);
      setPrediction(result);
      setActiveFindingIndex(null);
    } catch (requestError) {
      setError(requestError.message || "The analysis could not be started.");
    } finally {
      setIsPredicting(false);
    }
  };

  const handleResetDetection = () => {
    setSelectedFile(null);
    setPrediction(null);
    setActiveFindingIndex(null);
    setError("");
  };

  return (
    <div className="app-shell">
      <header className="clinic-header">
        <div className="brand-area">
          <img className="brand-logo" src={fractsureLogo} alt="Fractsure" />
          <div className="header-copy">
            <p className="eyebrow">Clinical Image Analysis</p>
            <h1>Fractsure Detection</h1>
            <p>Upload, AI detection, and result overview in one compact workflow.</p>
          </div>
        </div>
        <div className="project-info">
          <button className="info-trigger" type="button" aria-label="Project information">
            <Info size={17} />
          </button>
          <div className="project-popover" role="note">
            <strong>ZHAW</strong>
            <span>Machine Learning and Data in Operation</span>
            <span>Group No. 4: Fabio Scherrer, Simon Meyer, Gery Müller, Dario Eicher</span>
          </div>
        </div>
      </header>

      <main className="workspace" aria-label="Fracture Detection Workspace">
        <section className="upload-panel" aria-label="Image upload">
          <div className="panel-heading">
            <ShieldCheck size={18} />
            <div>
              <span className="step-label">Step 1</span>
              <h2>Prepare image</h2>
            </div>
          </div>

          <FileDropzone selectedFile={selectedFile} onFileSelect={handleFileSelect} />

          <button
            className="primary-action"
            type="button"
            onClick={handleRunDetection}
            disabled={!selectedFile || isPredicting}
          >
            <ScanSearch size={19} />
            <span>{isPredicting ? "Analyzing..." : "Start analysis"}</span>
          </button>
        </section>

        <PredictionViewer
          error={error}
          imageName={selectedFile?.name}
          imageUrl={imageUrl}
          isLoading={isPredicting}
          prediction={prediction}
          activeFindingIndex={activeFindingIndex}
          onActiveFindingChange={setActiveFindingIndex}
        />

        <ResultPanel
          prediction={prediction}
          activeFindingIndex={activeFindingIndex}
          onActiveFindingChange={setActiveFindingIndex}
          onResetDetection={handleResetDetection}
        />
      </main>
    </div>
  );
}

export default App;
