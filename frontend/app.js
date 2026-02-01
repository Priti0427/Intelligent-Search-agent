/**
 * Agentic Search Frontend Application
 * INFO 624: Intelligent Search and Language Models
 */

// Configuration
const API_BASE = '/api';
let settings = {
    includeSources: true,
    maxResults: 5,
};
let queryCount = 0;

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const settingsModal = document.getElementById('settingsModal');
const settingsBtn = document.getElementById('settingsBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadSettings();
    
    // Auto-resize textarea
    queryInput.addEventListener('input', autoResize);
    
    // Settings button
    settingsBtn.addEventListener('click', openSettings);
});

/**
 * Auto-resize textarea based on content
 */
function autoResize() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
}

/**
 * Handle keyboard events in input
 */
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuery();
    }
}

/**
 * Set query from example button
 */
function setQuery(query) {
    queryInput.value = query;
    autoResize();
    queryInput.focus();
}

/**
 * Send search query to API
 */
async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Disable input
    sendBtn.disabled = true;
    queryInput.disabled = true;
    
    // Add user message
    addMessage(query, 'user');
    queryInput.value = '';
    autoResize();
    
    // Add loading message
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                include_sources: settings.includeSources,
                max_results: settings.maxResults,
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Add assistant response
        addAssistantMessage(data);
        
        // Update stats
        queryCount++;
        document.getElementById('queryCount').textContent = queryCount;
        
    } catch (error) {
        console.error('Search error:', error);
        removeMessage(loadingId);
        addErrorMessage(error.message);
    } finally {
        sendBtn.disabled = false;
        queryInput.disabled = false;
        queryInput.focus();
    }
}

/**
 * Add a message to the chat
 */
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.innerHTML = `
        <div class="message-avatar">${type === 'user' ? '👤' : '🤖'}</div>
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

/**
 * Add loading message
 */
function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.id = id;
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="loading-message">
                <div class="loading-spinner"></div>
                <span>Searching and analyzing...</span>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

/**
 * Remove a message by ID
 */
function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

/**
 * Add assistant response with citations
 */
function addAssistantMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    
    let citationsHtml = '';
    if (data.citations && data.citations.length > 0) {
        citationsHtml = `
            <div class="citations">
                <div class="citations-title">Sources</div>
                ${data.citations.map(c => `
                    <div class="citation">
                        <div class="citation-number">${c.number}</div>
                        <div class="citation-content">
                            <div class="citation-title">${escapeHtml(c.title)}</div>
                            <div class="citation-type">${c.source_type}</div>
                            ${c.url ? `<a href="${escapeHtml(c.url)}" target="_blank" class="citation-url">${escapeHtml(c.url)}</a>` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    let qualityHtml = '';
    if (data.quality_scores) {
        const qs = data.quality_scores;
        qualityHtml = `
            <div class="quality-scores">
                <div class="quality-title">Quality Assessment</div>
                ${createScoreBar('Relevance', qs.relevance)}
                ${createScoreBar('Completeness', qs.completeness)}
                ${createScoreBar('Accuracy', qs.accuracy)}
                ${createScoreBar('Citations', qs.citation_quality)}
                ${createScoreBar('Clarity', qs.clarity)}
            </div>
        `;
    }
    
    let metadataHtml = '';
    if (data.metadata) {
        const m = data.metadata;
        metadataHtml = `
            <div class="metadata">
                <div class="metadata-item">
                    <span>Query Type:</span>
                    <span>${m.query_type}</span>
                </div>
                <div class="metadata-item">
                    <span>Sources:</span>
                    <span>${m.sources_searched.join(', ')}</span>
                </div>
                <div class="metadata-item">
                    <span>Results:</span>
                    <span>${m.total_results}</span>
                </div>
                <div class="metadata-item">
                    <span>Time:</span>
                    <span>${m.processing_time_ms.toFixed(0)}ms</span>
                </div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="answer-text">${formatAnswer(data.answer)}</div>
            ${citationsHtml}
            ${qualityHtml}
            ${metadataHtml}
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Create a score bar HTML
 */
function createScoreBar(label, value) {
    const percentage = (value * 100).toFixed(0);
    return `
        <div class="score-item">
            <span class="score-label">${label}</span>
            <div class="score-bar">
                <div class="score-fill" style="width: ${percentage}%"></div>
            </div>
        </div>
    `;
}

/**
 * Add error message
 */
function addErrorMessage(error) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">⚠️</div>
        <div class="message-content">
            <p style="color: var(--error);">Sorry, an error occurred: ${escapeHtml(error)}</p>
            <p>Please try again or check the API status.</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Format answer text with markdown-like formatting
 */
function formatAnswer(text) {
    if (!text) return '';
    
    // Escape HTML first
    let formatted = escapeHtml(text);
    
    // Convert markdown-style formatting
    formatted = formatted
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Code
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        // Citation references
        .replace(/\[Source (\d+)\]/g, '<sup class="citation-ref">[$1]</sup>');
    
    return `<p>${formatted}</p>`;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Check API health status
 */
async function checkHealth() {
    const statusContent = document.getElementById('statusContent');
    
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        let statusHtml = '';
        for (const service of data.services) {
            const statusClass = service.status === 'ok' ? 'ok' : 
                               service.status === 'error' ? 'error' : 'warning';
            statusHtml += `
                <div class="status-item">
                    <span class="status-dot ${statusClass}"></span>
                    <span>${service.name}: ${service.message || service.status}</span>
                </div>
            `;
        }
        
        statusContent.innerHTML = statusHtml;
        
        // Update document count
        document.getElementById('docCount').textContent = data.document_count;
        
    } catch (error) {
        statusContent.innerHTML = `
            <div class="status-item">
                <span class="status-dot error"></span>
                <span>API unavailable</span>
            </div>
        `;
    }
}

/**
 * Settings management
 */
function openSettings() {
    settingsModal.classList.remove('hidden');
    document.getElementById('includeSources').checked = settings.includeSources;
    document.getElementById('maxResults').value = settings.maxResults;
}

function closeSettings() {
    settingsModal.classList.add('hidden');
}

function saveSettings() {
    settings.includeSources = document.getElementById('includeSources').checked;
    settings.maxResults = parseInt(document.getElementById('maxResults').value) || 5;
    localStorage.setItem('agenticSearchSettings', JSON.stringify(settings));
    closeSettings();
}

function loadSettings() {
    const saved = localStorage.getItem('agenticSearchSettings');
    if (saved) {
        try {
            settings = JSON.parse(saved);
        } catch (e) {
            console.error('Failed to load settings:', e);
        }
    }
}

// Refresh health status periodically
setInterval(checkHealth, 60000);
