/**
 * predictor.js
 * 
 * This file separates the prediction logic from the UI code.
 * It connects to our local FastAPI python server to run ML predictions.
 */

// Module-level state to throttle requests. 
// We want ~3-5 API calls per second instead of spamming 30-60 frames per second.
let lastRequestTime = 0;
const THROTTLE_MS = 250; // 4 requests per second
let isRequestPending = false;

/**
 * Predicts the sign language gesture based on hand landmarks.
 * 
 * @param {Array} landmarks - The raw 21 hand landmarks provided by MediaPipe.
 * @returns {Promise<string|null>} - The predicted sign label based on the local FastAPI return, or null if skipped/error.
 */
export const predictSign = async (landmarks) => {
    // 1. Flatten the landmarks into a 1D numeric array [x1, y1, z1, x2, y2, z2, ...]
    // This format matches the 63 inputs our FastAPI server expects.
    const flattenedLandmarks = landmarks.flatMap(landmark => [landmark.x, landmark.y, landmark.z]);

    // Our MediaPipe hand has exactly 21 landmarks. 21 * 3 = 63.
    if (flattenedLandmarks.length !== 63) {
        return null;
    }

    const now = Date.now();
    // 2. Throttle API requests so we don't bombard the Python server.
    // Also, don't send a new request if the last one is still pending.
    if (now - lastRequestTime < THROTTLE_MS || isRequestPending) {
        return null; // Return null effectively meaning "ignore this frame"
    }

    // 3. Update our throttle variables
    lastRequestTime = now;
    isRequestPending = true;

    try {
        // 4. Send POST request to our FastAPI endpoint that executes the SciKit-Learn Model
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
        return data.prediction; // Returns the predicted sign string (e.g., "A", "B", "PEACE")
    } catch (error) {
        console.error("Failed to fetch prediction from FastAPI server:", error);
        return null;
    } finally {
        // Unlock to allow the next throttle period to succeed
        isRequestPending = false;
    }
};
