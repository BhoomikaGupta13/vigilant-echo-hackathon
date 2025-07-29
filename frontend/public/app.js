document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const resultsDiv = document.getElementById('results');
    // const refreshSourcesBtn = document.getElementById('refreshSourcesBtn'); // NEW: Button for sources list
    // const trackedSourcesList = document.getElementById('trackedSourcesList'); // NEW: Div for sources list

    // Helper function to map sentiment labels to human-readable strings
    function mapSentimentLabel(label) {
        switch (label) {
            case 'LABEL_0':
                return 'Negative 😠';
            case 'LABEL_1':
                return 'Neutral 😐';
            case 'LABEL_2':
                return 'Positive 😊';
            default:
                return label; // Return as-is if unrecognized
        }
    }

    // Helper functions for display classes (Trust Level)
    function getTrustLevel(data) {
        const cm_sdd = data.cm_sdd || {};
        const ls_zlf = data.ls_zlf || {};

        const isDiscrepant = cm_sdd.discrepancy_detected;
        const isDeepfakeDetected = ls_zlf.deepfake_analysis && ls_zlf.deepfake_analysis.deepfake_detected;
        const isLLMOriginNotHuman = ls_zlf.llm_origin_analysis && 
                                     ls_zlf.llm_origin_analysis.llm_origin !== "N/A" && 
                                     ls_zlf.llm_origin_analysis.llm_origin !== "Human/Uncertain";

        if (isDiscrepant || isDeepfakeDetected) {
            return "Low Trust";
        }
        if (isLLMOriginNotHuman) {
            return "Moderate Trust (AI-generated)";
        }
        return "High Trust";
    }

    function getTrustLevelClass(data) {
        const cm_sdd = data.cm_sdd || {};
        const ls_zlf = data.ls_zlf || {};

        const isDiscrepant = cm_sdd.discrepancy_detected;
        const isDeepfakeDetected = ls_zlf.deepfake_analysis && ls_zlf.deepfake_analysis.deepfake_detected;
        const isLLMOriginNotHuman = ls_zlf.llm_origin_analysis && 
                                     ls_zlf.llm_origin_analysis.llm_origin !== "N/A" && 
                                     ls_zlf.llm_origin_analysis.llm_origin !== "Human/Uncertain";

        if (isDiscrepant || isDeepfakeDetected) {
            return 'trust-low';
        }
        if (isLLMOriginNotHuman) {
            return 'trust-moderate';
        }
        return 'trust-high';
    }

    // Helper function for LLM Origin class (colors)
    function getLLMOriginClass(origin) {
        if (origin) {
            if (origin.includes("AI") || origin.includes("LLM") || origin.includes("Uncertain")) return 'alert-orange';
            if (origin.includes("Human")) return 'alert-green';
        }
        return '';
    }

    // Helper function for Source Risk display (Updated for nuanced levels)
    // function getSourceRiskClass(riskLevelText) {
    //     switch (riskLevelText) {
    //         case 'Low Risk':
    //             return 'risk-low';
    //         case 'Medium Risk':
    //             return 'risk-medium';
    //         case 'High Risk':
    //             return 'risk-high';
    //         case 'Critical Risk':
    //             return 'risk-critical';
    //         default:
    //             return '';
    //     }
    // }


    // Function to build the main analysis results HTML
    function buildResultsHTML(data) {
        const cm_sdd = data.cm_sdd || {};
        const ls_zlf = data.ls_zlf || {};
        const dp_cng_suggestion = data.dp_cng_suggestion || 'N/A';
        const source_tracking = data.source_tracking || {}; // Get source tracking data

        // Process discrepancy reasons for display
        const discrepancyReasonsHtml = cm_sdd.discrepancy_reason && cm_sdd.discrepancy_reason.length > 0
            ? `<ul>${cm_sdd.discrepancy_reason.map(reason => {
                // Apply sentiment mapping to the reason string if it contains labels
                let correctedReason = reason.replace(/LABEL_0/g, mapSentimentLabel('LABEL_0'))
                                            .replace(/LABEL_1/g, mapSentimentLabel('LABEL_1'))
                                            .replace(/LABEL_2/g, mapSentimentLabel('LABEL_2'));
                return `<li>${correctedReason}</li>`;
            }).join('')}</ul>`
            : `<em>No specific reasons found or not applicable.</em>`;

        return `
            <div class="results-container">
                <div class="result-header">
                    <h2>🔍 Analysis Results</h2>
                    <p class="overall-status"><strong>Overall Status:</strong> ${data.overall_status || "Completed."}</p>
                </div>

                <div class="analysis-grid">
                    <div class="result-card">
                        <h3>🔀 Cross-Modal Semantic Discrepancy (CM-SDD)</h3>
                        <div class="alert-box ${cm_sdd.discrepancy_detected ? 'alert-danger' : 'alert-success'}">
                            <strong>Detected Discrepancy:</strong> 
                            <span class="${cm_sdd.discrepancy_detected ? 'alert-red' : 'alert-green'}">
                                ${cm_sdd.discrepancy_detected ? 'Yes 🚨' : 'No ✅'}
                            </span>
                        </div>
                        
                        ${cm_sdd.discrepancy_detected ? // Only show reason box if discrepancy is detected
                            `<div class="reason-box">
                                <strong>Reason(s):</strong>
                                ${discrepancyReasonsHtml}
                            </div>` 
                            : '' // No reason box if no discrepancy
                        }
                        
                        <div class="details-section">
                            <h4>📊 Analysis Details:</h4>
                            <div class="detail-grid">
                                <div class="detail-item">
                                    <strong>Text Sentiment:</strong> 
                                    <em>${mapSentimentLabel(cm_sdd.text_sentiment.label || '')}</em> 
                                    (Score: ${cm_sdd.text_sentiment.score ? cm_sdd.text_sentiment.score.toFixed(2) : 'N/A'})
                                </div>
                                <div class="detail-item">
                                    <strong>Audio Transcript Sentiment:</strong> 
                                    <em>${mapSentimentLabel(cm_sdd.audio_sentiment_implied_by_transcript.label || '')}</em> 
                                    (Score: ${cm_sdd.audio_sentiment_implied_by_transcript.score ? cm_sdd.audio_sentiment_implied_by_transcript.score.toFixed(2) : 'N/A'})
                                </div>
                                <div class="detail-item">
                                    <strong>Text vs Audio Similarity:</strong> 
                                    ${cm_sdd.semantic_similarity_scores && cm_sdd.semantic_similarity_scores.text_audio ? cm_sdd.semantic_similarity_scores.text_audio.toFixed(2) : 'N/A'}
                                </div>
                            </div>
                            
                            <div class="content-previews">
                                <div class="preview-item">
                                    <strong>Original Text (Preview):</strong>
                                    <div class="content-snippet">${cm_sdd.text_content ? cm_sdd.text_content.substring(0, 150) + '...' : 'N/A'}</div>
                                </div>
                                <div class="preview-item">
                                    <strong>Audio Transcribed (Preview):</strong>
                                    <div class="content-snippet">${cm_sdd.audio_transcript ? cm_sdd.audio_transcript.substring(0, 150) + '...' : 'N/A'}</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="result-card">
                        <h3>🤖 Deepfake & LLM Fingerprint (LS-ZLF)</h3>
                        <div class="alert-box ${ls_zlf.deepfake_analysis && ls_zlf.deepfake_analysis.deepfake_detected ? 'alert-danger' : 'alert-success'}">
                            <strong>Deepfake Detected:</strong> 
                            <span class="${ls_zlf.deepfake_analysis && ls_zlf.deepfake_analysis.deepfake_detected ? 'alert-red' : 'alert-green'}">
                                ${ls_zlf.deepfake_analysis && ls_zlf.deepfake_analysis.deepfake_detected ? 'Yes 🚨' : 'No ✅'}
                            </span>
                        </div>
                        <div class="reason-box">
                            <strong>Reason:</strong> ${ls_zlf.deepfake_analysis && ls_zlf.deepfake_analysis.reason ? ls_zlf.deepfake_analysis.reason : 'N/A'}
                        </div>
                        
                        <div class="llm-analysis">
                            <div class="detail-item">
                                <strong>LLM Origin:</strong> 
                                <span class="${getLLMOriginClass(ls_zlf.llm_origin_analysis && ls_zlf.llm_origin_analysis.llm_origin)}">
                                    ${ls_zlf.llm_origin_analysis && ls_zlf.llm_origin_analysis.llm_origin ? ls_zlf.llm_origin_analysis.llm_origin : 'N/A'}
                                </span> 
                                (Confidence: ${ls_zlf.llm_origin_analysis && ls_zlf.llm_origin_analysis.confidence ? (ls_zlf.llm_origin_analysis.confidence * 100).toFixed(1) + '%' : 'N/A'})
                            </div>
                            <div class="reason-box">
                                <strong>Reason:</strong> ${ls_zlf.llm_origin_analysis && ls_zlf.llm_origin_analysis.reason ? ls_zlf.llm_origin_analysis.reason : 'N/A'}
                            </div>
                        </div>
                    </div>

                    <div class="result-card">
                        <h3>💬 Counter-Narrative Suggestion (DP-CNG)</h3>
                        <p><em>Our autonomous system suggests the following response:</em></p>
                        <div class="counter-message">${dp_cng_suggestion}</div>
                    </div>

                    <div class="result-card">
                        <h3>🕵️ Source Tracking & Flagging</h3>
                        <div class="detail-item">
                            <strong>Source ID:</strong> ${source_tracking.source_id || 'N/A'}
                        </div>
                        <div class="detail-item">
                            <strong>Flag Count:</strong> ${source_tracking.flag_count}
                        </div>
                        <div class="detail-item">
                            <strong>Risk Status:</strong> 
                            <span class="${getSourceRiskClass(source_tracking.risk_level_text)}">
                                ${source_tracking.risk_level_text || 'N/A'} ${source_tracking.risk_level_text === 'Low Risk' ? '🟢' : 
                                source_tracking.risk_level_text === 'Medium Risk' ? '🟡' : 
                                source_tracking.risk_level_text === 'High Risk' ? '🟠' : 
                                source_tracking.risk_level_text === 'Critical Risk' ? '🔴' : ''}
                            </span>
                        </div>
                        <div class="detail-item">
                            <strong>Tracking Status:</strong> ${source_tracking.status}
                        </div>
                        <p class="small-text"><em>Source flags are based on internal system analysis. 'High Risk' indicates multiple flagged contents.</em></p>
                    </div>
                    <div class="result-card trust-card">
                        <h3>🛡️ Explainable Alert & Trust (EA-TD)</h3>
                        <div class="trust-level-box">
                            <strong>Estimated Trust Level:</strong> 
                            <span class="trust-badge ${getTrustLevelClass(data)}">${getTrustLevel(data)}</span>
                        </div>
                        
                        <div class="explanation-section">
                            <h4>🔍 How we know:</h4>
                            <p>Our system processed the content using multiple analysis modules:</p>
                            <ul>
                                <li>The <strong>Cross-Modal Semantic Discrepancy Detector (CM-SDD)</strong> identified core inconsistencies across media types (e.g., text vs. audio sentiment/meaning).</li>
                                <li>The <strong>Latent-Space Zero-Shot Deepfake & LLM Fingerprint Analyzer (LS-ZLF)</strong> assessed the content for deepfake patterns and identified the likely origin of the text.</li>
                                <li>Based on these findings, a tailored counter-narrative was suggested.</li>
                            </ul>
                            <p>This approach ensures our system is <strong>adversarial-resilient</strong> by looking for novel patterns, and provides <strong>explainable alerts</strong> that you can trust.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Event listener for form submission
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Stop default form submission

        // Add loading state to resultsDiv
        resultsDiv.innerHTML = '<p class="loading-message">Analyzing content... Please wait. This may take a moment as models process.</p>';
        resultsDiv.className = 'loading'; // Add loading class (you'll need CSS for this)

        const formData = new FormData();
        
        const videoFile = document.getElementById('videoFile').files[0];
        const audioFile = document.getElementById('audioFile').files[0];
        const textFile = document.getElementById('textFile').files[0];
        const sourceId = document.getElementById('sourceId').value; // Get source ID value

        // Debug logs for frontend file objects
        console.log('Frontend Debug: videoFile object:', videoFile);
        console.log('Frontend Debug: audioFile object:', audioFile);
        console.log('Frontend Debug: textFile object:', textFile);
        console.log('Frontend Debug: sourceId value:', sourceId);

        // Only append files that are actually selected
        if (videoFile) formData.append('video', videoFile);
        if (audioFile) formData.append('audio', audioFile);
        if (textFile) formData.append('text', textFile);
        if (sourceId) formData.append('source_id', sourceId); // Append source ID if provided

        // Basic validation: ensure at least one file is selected
        if (!videoFile && !audioFile && !textFile) {
            resultsDiv.innerHTML = '<p class="error">Please upload at least one file for analysis.</p>';
            resultsDiv.className = 'error-state';
            return;
        }

        try {
            // IMPORTANT: Ensure your backend is running on http://127.0.0.1:8000
            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                body: formData, // FormData automatically sets Content-Type to multipart/form-data
            });

            const data = await response.json(); // Parse the JSON response
            console.log("Backend Response:", data); // Log for debugging

            if (response.ok) {
                // Clear loading class and display results
                resultsDiv.className = 'results-display';
                
                // Build the HTML structure properly
                resultsDiv.innerHTML = buildResultsHTML(data);

            } else {
                resultsDiv.innerHTML = `<p class="error">Error during analysis: ${data.detail || 'Unknown error'}</p>`;
                resultsDiv.className = 'error-state';
            }
        } catch (error) {
            console.error('Network or parsing error:', error);
            resultsDiv.innerHTML = `<p class="error">An error occurred while connecting to the server. Is the backend running?</p>`;
            resultsDiv.className = 'error-state';
        }
    });

    // NEW: Event listener for refresh sources button (added outside DOMContentLoaded listener)
    // No, this should be INSIDE DOMContentLoaded to ensure elements exist.
    // Moved it inside below this comment block.
    
    // Moved inside DOMContentLoaded below
    // if (refreshSourcesBtn) { 
    //     refreshSourcesBtn.addEventListener('click', async () => {
    //         trackedSourcesList.innerHTML = '<p class="loading-message">Loading tracked sources...</p>';
    //         try {
    //             const response = await fetch('http://127.0.0.1:8000/sources');
    //             const sources = await response.json();
    //             console.log("Tracked Sources Backend Response:", sources); // Debug

    //             if (response.ok) {
    //                 if (sources.length === 0) {
    //                     trackedSourcesList.innerHTML = '<p class="initial-message">No sources tracked yet. Analyze some content first!</p>';
    //                 } else {
    //                     // Build HTML list of sources
    //                     let sourcesHtml = '<h3>All Tracked Sources:</h3><div class="sources-grid">';
    //                     sources.forEach(source => {
    //                         // Calculate risk_level_text on frontend for consistency
    //                         let currentRiskLevelText = "Low Risk";
    //                         const flags = source.flag_count || 0;
    //                         if (flags >= 5) currentRiskLevelText = "Critical Risk"; // Must match backend CRITICAL_RISK_THRESHOLD
    //                         else if (flags >= 3) currentRiskLevelText = "High Risk"; // Must match backend HIGH_RISK_THRESHOLD
    //                         else if (flags >= 1) currentRiskLevelText = "Medium Risk"; // Must match backend MEDIUM_RISK_THRESHOLD

    //                         sourcesHtml += `
    //                             <div class="source-card">
    //                                 <h4>${source.source_id}</h4>
    //                                 <p><strong>Flags:</strong> ${source.flag_count}</p>
    //                                 <p><strong>Risk:</strong> <span class="${getSourceRiskClass(currentRiskLevelText)}">${currentRiskLevelText}</span></p>
    //                                 <p class="small-text">Last flagged: ${source.last_flagged_at ? new Date(source.last_flagged_at).toLocaleString() : 'N/A'}</p>
    //                             </div>
    //                         `;
    //                     });
    //                     sourcesHtml += '</div>';
    //                     trackedSourcesList.innerHTML = sourcesHtml;
    //                 }
    //             } else {
    //                 trackedSourcesList.innerHTML = `<p class="error">Error loading sources: ${sources.detail || 'Unknown error'}</p>`;
    //             }
    //         } catch (error) {
    //             console.error('Network error fetching sources:', error);
    //             trackedSourcesList.innerHTML = `<p class="error">Could not connect to backend to fetch sources.</p>`;
    //         }
    //     });
    // }

}); // End of DOMContentLoaded event listener