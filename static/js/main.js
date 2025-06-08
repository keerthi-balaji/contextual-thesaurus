let modelLoaded = false;
let checkCount = 0;
const MAX_RETRIES = 10;

async function checkModelStatus() {
    const loadingDiv = document.getElementById('loading');
    if (checkCount === 0) {
        loadingDiv.innerHTML = `
            <div class="spinner"></div>
            <p>Loading language models... This may take a few minutes on first run.</p>
            <p class="loading-detail">Downloading BERT model and GloVe embeddings...</p>
        `;
    }
    
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        if (data.error) {
            loadingDiv.innerHTML = `
                <div class="error">
                    <p>Error loading models:</p>
                    <p class="error-detail">${data.error}</p>
                    <button onclick="window.location.reload()">Retry</button>
                </div>
            `;
            return;
        }
        
        modelLoaded = data.ready;
        if (modelLoaded) {
            loadingDiv.classList.add('hidden');
            document.getElementById('input-controls').classList.remove('hidden');
        } else {
            checkCount++;
            if (checkCount >= MAX_RETRIES) {
                loadingDiv.innerHTML = `
                    <div class="error">
                        <p>Timeout loading models. Please refresh the page to try again.</p>
                        <button onclick="window.location.reload()">Refresh</button>
                    </div>
                `;
                return;
            }
            // Check again in 3 seconds
            setTimeout(checkModelStatus, 3000);
        }
    } catch (error) {
        loadingDiv.innerHTML = `
            <div class="error">
                <p>Error connecting to server:</p>
                <p class="error-detail">${error.message}</p>
                <button onclick="window.location.reload()">Retry</button>
            </div>
        `;
    }
}

async function getSuggestions() {
    if (!modelLoaded) {
        alert('Please wait for models to finish loading');
        return;
    }

    const sentence = document.getElementById('sentence').value;
    const word = document.getElementById('word').value;
    
    if (!sentence || !word) {
        alert('Please enter both a sentence and a word');
        return;
    }

    const loadingDiv = document.getElementById('loading');
    loadingDiv.innerHTML = `
        <div class="spinner"></div>
        <p>Generating suggestions...</p>
    `;
    loadingDiv.classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');

    try {
        const response = await fetch('/suggest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sentence: sentence,
                word: word,
                top_n: 5
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displaySuggestions(data.suggestions);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = `
            <div class="error">
                Error getting suggestions: ${error.message}
                <br>
                Please try again.
            </div>
        `;
        document.getElementById('results').classList.remove('hidden');
    } finally {
        loadingDiv.classList.add('hidden');
    }
}

function displaySuggestions(suggestions) {
    const container = document.getElementById('suggestions');
    container.innerHTML = '';

    suggestions.forEach(suggestion => {
        const card = document.createElement('div');
        card.className = 'suggestion-card';
        card.innerHTML = `
            <h3>${suggestion.word}</h3>
            <div class="score-details">
                Overall Score: ${(suggestion.score * 100).toFixed(1)}%
                <br>
                GloVe Score: ${(suggestion.glove_component * 100).toFixed(1)}%
                <br>
                BERT Context Score: ${(suggestion.bert_component * 100).toFixed(1)}%
            </div>
            <div class="definition">
                ${suggestion.definition}
            </div>
        `;
        container.appendChild(card);
    });

    document.getElementById('results').classList.remove('hidden');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', checkModelStatus);