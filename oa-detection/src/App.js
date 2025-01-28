import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null); // Image URL for displaying the image after prediction
  const [inputKey, setInputKey] = useState(Date.now()); // Unique key to force file input re-render
  const [error, setError] = useState(null); // Add error state

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setPrediction(null); // Reset the prediction when a new file is uploaded
    setPreviewUrl(URL.createObjectURL(selectedFile)); // Set preview URL immediately on file upload
    setError(null); // Clear any previous errors
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      setError(null); // Clear any previous errors
      const response = await axios.post('http://localhost:8000/api/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data); // Store the entire prediction response (grade and probabilities)
    } catch (error) {
      console.error("Error uploading the file or getting the prediction.", error);
      setError(error.response?.data?.error || "Error uploading file or making prediction.");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null); // Clear the file
    setPrediction(null); // Clear the prediction
    setPreviewUrl(null); // Clear the preview
    setInputKey(Date.now()); // Reset the file input by changing its key
    setError(null); // Clear any errors on reset
  };

  return (
    <div className="App">
      <div className="card">
        <h1 className="title">Knee Osteoarthritis Severity Classification</h1>
        {!prediction ? (
          <form onSubmit={handleSubmit}>
            <input
              key={inputKey} // Key is updated to reset the file input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="file-input"
            />
            <button type="submit" className="btn" disabled={loading}>
              {loading ? "Predicting..." : "Predict Grade"}
            </button>
            {/* Display error message if exists */}
            {error && <div className="error-message">{error}</div>}
          </form>
        ) : (
          <div className="result">
            {previewUrl && <img src={previewUrl} alt="Predicted Image" className="image-preview" />}
            <h2>Prediction Result</h2>
            <p><strong>{prediction.prediction}</strong></p> {/* Display predicted grade */}
            <h3>Probabilities:</h3>
            <div className="probabilities-container">
              {prediction.probabilities.map((prob, index) => (
                <div className="probability-row" key={index}>
                  <span className="class-label">Class {prob.class}:</span>
                  <span className="class-probability">{prob.probability}</span>
                </div>
              ))}
            </div>
            <button onClick={handleReset} className="btn-reset">Predict Another Image</button>
          </div>
        )}

        {loading && <p className="loading-text">Predicting...</p>}
      </div>
    </div>
  );
}

export default App;
