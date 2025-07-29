import { defineConfig } from 'vite'

export default defineConfig({
  root: '.', // optional, default is .
})


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
                // Update resultsDiv with initial success message or basic data
                resultsDiv.innerHTML = `<p style="color: green;">${data.overall_status || "Analysis initiated successfully!"}</p>`;
                resultsDiv.style.justifyContent = 'flex-start'; // Align text left
                // You'll enhance this display significantly in later steps
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
});