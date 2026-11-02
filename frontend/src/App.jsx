import { useRef, useState } from "react";
import "./App.css";

// Address of the FastAPI server.
const API_URL = "http://127.0.0.1:8000/predict";

// Colour level of a confidence value: high (>= 80%), mid (>= 50%), low (< 50%).
function level(conf) {
  if (conf >= 0.8) return "high";
  if (conf >= 0.5) return "mid";
  return "low";
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

function CarIcon({ size = 22 }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor"
         strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 13l2-5a2 2 0 0 1 1.9-1.3h8.2A2 2 0 0 1 17 8l2 5" />
      <rect x="2" y="13" width="20" height="5" rx="2" />
      <circle cx="7" cy="18" r="1.6" />
      <circle cx="17" cy="18" r="1.6" />
    </svg>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  // Store the chosen image and show a preview.
  function chooseFile(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  }

  // Go back to the empty state.
  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  // Send the image to the API and show the answer.
  async function predict() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(API_URL, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (e) {
      setError(
        e.message === "Failed to fetch"
          ? "Cannot reach the API. Is the server running?"
          : e.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="logo"><CarIcon /></div>
        <div>
          <h1>Car Brand Classifier</h1>
          <p className="subtitle">Upload a car photo and the model names the brand.</p>
        </div>
      </header>

      <div className="chips">
        <span className="chip">20 brands</span>
        <span className="chip">ResNet18</span>
        <span className="chip">88% test accuracy</span>
      </div>

      <main className="card">
        <div
          className={`drop ${dragging ? "drag" : ""} ${preview ? "filled" : ""}`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            chooseFile(e.dataTransfer.files[0]);
          }}
        >
          {preview ? (
            <img src={preview} alt="Selected car" className="preview" />
          ) : (
            <div className="placeholder">
              <CarIcon size={40} />
              <p className="drop-title">Drop a car photo here</p>
              <p className="drop-hint">or click to browse · JPEG, PNG or WebP · max 8 MB</p>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            hidden
            onChange={(e) => chooseFile(e.target.files[0])}
          />
        </div>

        <div className="actions">
          <button className="btn" onClick={predict} disabled={!file || loading}>
            {loading ? <><span className="spinner" /> Analysing...</> : "Predict brand"}
          </button>
          {file && !loading && (
            <button className="btn ghost" onClick={reset}>Choose another</button>
          )}
        </div>

        {error && <div className="error">{error}</div>}

                {result && result.is_car === false && (
          <div className="warn big">{result.message}</div>
        )}

        {result && result.is_car !== false && (
          <section className={`result ${level(result.confidence)}`}>
            <div className="label">Predicted brand</div>
            <div className="brand-row">
              <div className="brand">{result.predicted_class}</div>
              <div className="pill">{pct(result.confidence)}</div>
            </div>
            <div className="meter">
              <div className="meter-fill" style={{ width: `${result.confidence * 100}%` }} />
            </div>

            {result.warning && <div className="warn">{result.warning}</div>}

            <div className="label top">Top 3 candidates</div>
            {result.top3.map((t, i) => (
              <div className={`row ${i === 0 ? "first" : ""}`} key={t.class}>
                <span className="rank">{i + 1}</span>
                <span className="name">{t.class}</span>
                <div className="bar">
                  <div className="fill" style={{ width: `${t.confidence * 100}%` }} />
                </div>
                <span className="pct">{pct(t.confidence)}</span>
              </div>
            ))}
          </section>
        )}
      </main>

      <footer className="footer">
        Trained on Stanford Cars (20 brands). Other brands or non-car photos can still be
        assigned to one of these 20, so treat low confidence with care.
      </footer>
    </div>
  );
}