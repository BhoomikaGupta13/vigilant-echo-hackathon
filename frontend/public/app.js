
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const resultsDiv = document.getElementById('results');

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Stop default form submission

        resultsDiv.innerHTML = '<p>Analyzing content... Please wait.</p>';
        resultsDiv.style.justifyContent = 'flex-start'; // Align text left

        const formData = new FormData();
        
        const videoFile = document.getElementById('videoFile').files[0];
        const audioFile = document.getElementById('audioFile').files[0];
        const textFile = document.getElementById('textFile').files[0];

        console.log('Frontend Debug: videoFile object:', videoFile);
        console.log('Frontend Debug: audioFile object:', audioFile);
        console.log('Frontend Debug: textFile object:', textFile);
        // Only append files that are actually selected
        if (videoFile) formData.append('video', videoFile);
        if (audioFile) formData.append('audio', audioFile);
        if (textFile) formData.append('text', textFile);

        if (!videoFile && !audioFile && !textFile) {
            resultsDiv.innerHTML = '<p style="color: red;">Please upload at least one file for analysis.</p>';
            resultsDiv.style.justifyContent = 'center'; // Center error message
            return;
        }

        try {
            // IMPORTANT: Ensure your backend is running on http://127.0.0.1:8000
            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json(); // Parse the JSON response
            console.log("Backend Response:", data); // Log for debugging

            if (response.ok) {
                // --- FULL PHASE 3 DISPLAY LOGIC ---
                resultsDiv.innerHTML = `
                    <div class="result-section">
                        <h2>Analysis Results</h2>
                        <p><strong>Overall Status:</strong> ${data.overall_status || "Completed."}</p>
                    </div>

                    <div class="result-section">
                        <h3>Cross-Modal Semantic Discrepancy (CM-SDD)</h3>
                        <p><strong>Detected Discrepancy:</strong> <span class="${data.cm_sdd.discrepancy_detected ? 'alert-red' : 'alert-green'}">${data.cm_sdd.discrepancy_detected ? 'Yes 🚨' : 'No ✅'}</span></p>
                        ${data.cm_sdd.discrepancy_detected && data.cm_sdd.discrepancy_reason.length > 0 ? 
                            `<p><strong>Reason(s):</strong> <ul>${data.cm_sdd.discrepancy_reason.map(r => `<li>${r}</li>`).join('')}</ul></p>` 
                            : data.cm_sdd.discrepancy_detected ? `<p><strong>Reason(s):</strong> <em>No specific reasons provided by backend yet.</em></p>` : ''
                        }
                        
                        <h4>Details:</h4>
                        <p><strong>Text Sentiment:</strong> ${data.cm_sdd.text_sentiment.label || 'N/A'} (Score: ${data.cm_sdd.text_sentiment.score ? data.cm_sdd.text_sentiment.score.toFixed(2) : 'N/A'})</p>
                        <p><strong>Audio Transcript Sentiment:</strong> ${data.cm_sdd.audio_sentiment_implied_by_transcript.label || 'N/A'} (Score: ${data.cm_sdd.audio_sentiment_implied_by_transcript.score ? data.cm_sdd.audio_sentiment_implied_by_transcript.score.toFixed(2) : 'N/A'})</p>
                        <p><strong>Text vs Audio Similarity:</strong> ${data.cm_sdd.semantic_similarity_scores.text_audio ? data.cm_sdd.semantic_similarity_scores.text_audio.toFixed(2) : 'N/A'}</p>
                        <p><strong>Original Text (Snippet):</strong> <em>${data.cm_sdd.text_content ? data.cm_sdd.text_content.substring(0, 150) + '...' : 'N/A'}</em></p>
                        <p><strong>Audio Transcribed (Snippet):</strong> <em>${data.cm_sdd.audio_transcript ? data.cm_sdd.audio_transcript.substring(0, 150) + '...' : 'N/A'}</em></p>
                    </div>

                    <div class="result-section">
                        <h3>Deepfake & LLM Fingerprint (LS-ZLF)</h3>
                        <p><strong>Deepfake Detected:</strong> <span class="${data.ls_zlf.deepfake_analysis.deepfake_detected ? 'alert-red' : 'alert-green'}">${data.ls_zlf.deepfake_analysis.deepfake_detected ? 'Yes 🚨' : 'No ✅'}</span></p>
                        <p><strong>Reason:</strong> ${data.ls_zlf.deepfake_analysis.reason || 'N/A'}</p>
                        
                        <p><strong>LLM Origin:</strong> <span class="${getLLMOriginClass(data.ls_zlf.llm_origin_analysis.llm_origin)}">${data.ls_zlf.llm_origin_analysis.llm_origin || 'N/A'}</span> (Confidence: ${data.ls_zlf.llm_origin_analysis.confidence ? data.ls_zlf.llm_origin_analysis.confidence.toFixed(1) : 'N/A'})</p>
                        <p><strong>Reason:</strong> ${data.ls_zlf.llm_origin_analysis.reason || 'N/A'}</p>
                    </div>

                    <div class="result-section">
                        <h3>Counter-Narrative Suggestion (DP-CNG)</h3>
                        <p><em>Our autonomous system suggests the following response:</em></p>
                        <p class="counter-message">${data.dp_cng_suggestion || 'N/A'}</p>
                    </div>

                    <div class="result-section">
                        <h3>Explainable Alert & Trust (EA-TD)</h3>
                        <p><strong>Estimated Trust Level:</strong> <span class="${getTrustLevelClass(data)}">${getTrustLevel(data)}</span></p>
                        <p><strong>How we know:</strong></p>
                        <p>Our system processed the content. The <strong>Cross-Modal Semantic Discrepancy Detector (CM-SDD)</strong> identified core inconsistencies across media types (e.g., text vs. audio sentiment/meaning). The <strong>Latent-Space Zero-Shot Deepfake & LLM Fingerprint Analyzer (LS-ZLF)</strong> then assessed the content for deepfake patterns and identified the likely origin of the text. Based on these findings, a tailored counter-narrative was suggested.</p>
                        <p>This approach ensures our system is <strong>adversarial-resilient</strong> by looking for novel patterns, and provides <strong>explainable alerts</strong> that you can trust.</p>
                    </div>
                `;
                resultsDiv.style.justifyContent = 'flex-start'; // Align text left

            } else {
                resultsDiv.innerHTML = `<p style="color: red;">Error during analysis: ${data.detail || 'Unknown error'}</p>`;
                resultsDiv.style.justifyContent = 'center'; // Center error message
            }
        } catch (error) {
            console.error('Network or parsing error:', error);
            resultsDiv.innerHTML = `<p style="color: red;">An error occurred while connecting to the server. Is the backend running?</p>`;
            resultsDiv.style.justifyContent = 'center'; // Center error message
        }
    });

    // --- Helper functions for display ---
    function getTrustLevel(data) {
        // Accessing nested properties with optional chaining or logical AND for safety
        const isDeepfakeDetected = data.ls_zlf && data.ls_zlf.deepfake_analysis && data.ls_zlf.deepfake_analysis.deepfake_detected;
        const isLLMOriginNotHuman = data.ls_zlf && data.ls_zlf.llm_origin_analysis && 
                                     data.ls_zlf.llm_origin_analysis.llm_origin !== "N/A" && 
                                     data.ls_zlf.llm_origin_analysis.llm_origin !== "Human-Written";

        if (data.cm_sdd.discrepancy_detected || isDeepfakeDetected) {
            return "Low Trust";
        }
        if (isLLMOriginNotHuman) {
            return "Moderate Trust (AI-generated)";
        }
        return "High Trust";
    }

    function getTrustLevelClass(data) {
        const isDeepfakeDetected = data.ls_zlf && data.ls_zlf.deepfake_analysis && data.ls_zlf.deepfake_analysis.deepfake_detected;
        const isLLMOriginNotHuman = data.ls_zlf && data.ls_zlf.llm_origin_analysis && 
                                     data.ls_zlf.llm_origin_analysis.llm_origin !== "N/A" && 
                                     data.ls_zlf.llm_origin_analysis.llm_origin !== "Human-Written";

        if (data.cm_sdd.discrepancy_detected || isDeepfakeDetected) {
            return 'alert-red';
        }
        if (isLLMOriginNotHuman) {
            return 'alert-orange';
        }
        return 'alert-green';
    }

    function getLLMOriginClass(origin) {
        if (origin) {
            if (origin.includes("AI") || origin.includes("LLM") || origin.includes("Uncertain")) return 'alert-orange';
            if (origin.includes("Human")) return 'alert-green';
        }
        return ''; // No specific class
    }

});