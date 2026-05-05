import { useRef, useState } from "react";
import { FileImage, UploadCloud } from "lucide-react";

const acceptedTypes = ["image/png", "image/jpeg", "image/jpg"];

function isAcceptedImage(file) {
  return acceptedTypes.includes(file.type) || /\.(png|jpe?g)$/i.test(file.name);
}

export function FileDropzone({ selectedFile, onFileSelect }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const chooseFile = (file) => {
    if (file && isAcceptedImage(file)) {
      onFileSelect(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  };

  const handleInputChange = (event) => {
    chooseFile(event.target.files?.[0]);
    event.target.value = "";
  };

  return (
    <div
      className={`dropzone${isDragging ? " is-dragging" : ""}`}
      onDragEnter={() => setIsDragging(true)}
      onDragLeave={() => setIsDragging(false)}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        className="file-input"
        type="file"
        accept="image/png,image/jpeg"
        onChange={handleInputChange}
      />

      <div className="dropzone-icon" aria-hidden="true">
        <UploadCloud size={28} />
      </div>

      <div className="dropzone-copy">
        <strong>Select X-ray image</strong>
        <span>PNG or JPG</span>
      </div>

      <button className="secondary-action" type="button" onClick={() => inputRef.current?.click()}>
        <FileImage size={18} />
        <span>Choose file</span>
      </button>

      {selectedFile && (
        <div className="selected-file" title={selectedFile.name}>
          {selectedFile.name}
        </div>
      )}
    </div>
  );
}
