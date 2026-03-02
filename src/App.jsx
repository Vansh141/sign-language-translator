import { useEffect, useRef, useState } from 'react';
import { Camera } from '@mediapipe/camera_utils';
import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import './App.css';

function App() {
  // References to the video and canvas elements in the DOM
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // State for handling camera errors and tracking hand status
  const [error, setError] = useState(null);
  const [handDetected, setHandDetected] = useState(false);
  const [predictedSign, setPredictedSign] = useState("Waiting..."); // Default value as requested
  const [confidence, setConfidence] = useState(null); // Confidence level
  const [isLoading, setIsLoading] = useState(false); // API loading state

  // Refs for throttling API calls locally within the component
  const lastRequestTime = useRef(0);
  const isRequestPending = useRef(false);

  // Ref for prediction smoothing buffer (store last 3 predictions)
  const predictionBuffer = useRef([]);

  // Define prediction logic directly in App.jsx to better manage React states
  const predictSign = async (landmarks) => {
    try {
      // 1. Flatten the landmarks into a 1D numeric array [x1, y1, z1, x2, y2, z2, ...]
      const flattenedLandmarks = landmarks.flatMap(landmark => [landmark.x, landmark.y, landmark.z]);

      // Ensure exactly 63 values
      if (flattenedLandmarks.length !== 63) return;

      const now = Date.now();
      // Throttle requests to ~4 calls per second (250ms) and prevent overlapping
      if (now - lastRequestTime.current < 250 || isRequestPending.current) {
        return;
      }

      lastRequestTime.current = now;
      isRequestPending.current = true;
      setIsLoading(true);

      // Send POST request to FastAPI backend
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          landmarks: flattenedLandmarks
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      if (data && data.prediction) {
        // 1. Add new prediction to buffer
        predictionBuffer.current.push(data.prediction);

        // 2. Keep buffer size limited to 3
        if (predictionBuffer.current.length > 3) {
          predictionBuffer.current.shift();
        }

        // 3. Compute the most frequent label in the buffer
        const frequencyMap = {};
        let maxCount = 0;
        let smoothedPrediction = data.prediction; // fallback

        for (const sign of predictionBuffer.current) {
          frequencyMap[sign] = (frequencyMap[sign] || 0) + 1;
          if (frequencyMap[sign] > maxCount) {
            maxCount = frequencyMap[sign];
            smoothedPrediction = sign;
          }
        }

        // 4. Update state with the smoothed result
        setPredictedSign(smoothedPrediction);

        // 5. Update confidence
        // Extract confidence from API or gracefully fall back
        let currentConf = data.confidence !== undefined ? data.confidence : null;
        if (currentConf !== null) {
          // Normalize to a percentage if backend returns a fraction between 0.0 and 1.0
          if (currentConf <= 1 && currentConf > 0) currentConf = currentConf * 100;
          setConfidence(Math.round(currentConf));
        } else {
          setConfidence(null);
        }
      }
    } catch (err) {
      console.error("Failed to fetch prediction:", err);
      // Handle error state gracefully by updating the prediction value
      setPredictedSign("Error retrieving prediction");
    } finally {
      isRequestPending.current = false;
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;

    // Ensure both elements are available before starting
    if (!videoElement || !canvasElement) return;

    // Get the canvas context for drawing the landmarks
    const canvasCtx = canvasElement.getContext('2d');

    // Initialize the MediaPipe Hands model
    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    // Configure the hand tracking settings
    hands.setOptions({
      maxNumHands: 1, // We only want to detect a single hand
      modelComplexity: 1, // Higher complexity = more accurate but slower
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    // Handle results returned from the MediaPipe model each frame
    hands.onResults((results) => {
      // Clear the canvas from the previous frame before drawing new data
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      // Check if any hands are detected in the current frame
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        setHandDetected(true);

        // Loop through the detected hands and draw onto the canvas
        for (const landmarks of results.multiHandLandmarks) {
          drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
            color: '#00FF00', // Green lines connecting the joints
            lineWidth: 3
          });
          drawLandmarks(canvasCtx, landmarks, {
            color: '#FF0000', // Red dots for the joints
            lineWidth: 1,
            radius: 3
          });

          // Execute async prediction pipeline based on drawn hand
          predictSign(landmarks);
        }
      } else {
        // No hand found in this frame
        setHandDetected(false);
        setPredictedSign("Waiting..."); // Set back to default value when no hand
        setConfidence(null); // Reset confidence
        setIsLoading(false); // Reset loading
        predictionBuffer.current = []; // Clear buffer to prevent carrying over old predictions
      }
    });

    // Use MediaPipe's Camera utility to intelligently send webcam frames to the Hand model
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        // Send the current video frame to MediaPipe to find hands
        if (videoElement && videoElement.videoWidth && videoElement.videoHeight) {
          // Keep canvas internal resolution completely in-sync with real video resolution
          if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
          }
          try {
            await hands.send({ image: videoElement });
          } catch (err) {
            console.error("Error processing video frame: ", err);
          }
        }
      },
      width: 640,
      height: 480
    });

    // Start the camera and handle any permission errors
    camera.start().catch((err) => {
      console.error("Error accessing the camera: ", err);
      setError("Unable to access the camera. Please check your permissions.");
    });

    // Cleanup resources gracefully when the app is closed or restarted
    return () => {
      if (camera) {
        camera.stop();
      }
      if (hands) {
        hands.close();
      }
    };
  }, []); // Run effect only once on mount

  return (
    <div className="app-wrapper">
      {/* Top Navigation Bar */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">Sign Language Translator</div>
          <ul className="nav-links">
            <li><a href="#home">Home</a></li>
            <li><a href="#how-it-works">How It Works</a></li>
            <li><a href="#about">About</a></li>
          </ul>
        </div>
      </nav>

      {/* Main Container */}
      <div className="app-container" id="home">
        <header className="app-header">
          <h1>Real-time Pattern Recognition</h1>
          <p className="subtitle">Translate gestures to text natively in the browser</p>
        </header>

        <main className="main-content">
          {error ? (
            <div className="error-message">
              <p>{error}</p>
            </div>
          ) : (
            <div className="dashboard-wrapper">
              {/* Premium Dashboard Frame */}
              <div className="dashboard-card">
                <div className="video-container">
                  {/* The video element grabs the raw, natural webcam feed */}
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="webcam-video"
                  />
                  {/* The canvas sits cleanly above the video where MediaPipe draws tracked landmarks */}
                  <canvas
                    ref={canvasRef}
                    className="webcam-canvas"
                  />
                </div>

                {/* Clean Prediction Panel integrated inside Dashboard */}
                <div className="prediction-panel">
                  {/* Live Status Badge */}
                  <div className={`status-badge ${!handDetected ? 'status-red' : (isLoading && predictedSign === "Waiting..." ? 'status-yellow pulse' : 'status-green')}`}>
                    {!handDetected
                      ? "🔴 No Hand Detected"
                      : (isLoading && predictedSign === "Waiting..." ? "🟡 Predicting..." : "🟢 Hand Detected")}
                  </div>

                  {/* Data Content */}
                  <div className="prediction-content">
                    <h3 className="prediction-title">Predicted Sign</h3>
                    <div className="prediction-value">
                      {predictedSign === "Waiting..." ? "..." : predictedSign}
                    </div>
                    <div className="confidence-display" style={{
                      color: confidence ? (confidence > 85 ? '#4ade80' : confidence >= 65 ? '#fbbf24' : '#f87171') : 'inherit',
                      fontWeight: 'bold'
                    }}>
                      Confidence: {confidence ? `${confidence}%` : "--"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>

        {/* How It Works Section */}
        <section id="how-it-works" className="info-section">
          <h2>How It Works</h2>
          <div className="steps-container">
            <div className="step-card">
              <div className="step-number">1</div>
              <h3>Camera Capture</h3>
              <p>Your webcam captures your hand movements seamlessly.</p>
            </div>
            <div className="step-card">
              <div className="step-number">2</div>
              <h3>Landmark Tracking</h3>
              <p>MediaPipe tracks 21 unique 3D points on your hand.</p>
            </div>
            <div className="step-card">
              <div className="step-number">3</div>
              <h3>AI Prediction</h3>
              <p>A trained machine learning model analyzes the coordinates.</p>
            </div>
            <div className="step-card">
              <div className="step-number">4</div>
              <h3>Real-time Output</h3>
              <p>The translated sign is displayed to you instantly.</p>
            </div>
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="info-section">
          <h2>About The Project</h2>
          <div className="about-card">
            <p>This premium Sign Language Recognition application is designed to demonstrate how powerful machine learning models can run inference locally while connecting to robust remote backend engines.</p>
            <div className="tech-stack">
              <h4>Tech Stack Used</h4>
              <div className="tech-tags">
                <span className="tech-tag">React</span>
                <span className="tech-tag">MediaPipe</span>
                <span className="tech-tag">FastAPI</span>
                <span className="tech-tag">scikit-learn</span>
                <span className="tech-tag">Python</span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
