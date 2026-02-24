/**
 * Hygiene Check Dashboard - Chart.js wrappers and score gauges
 */

/**
 * Draw a doughnut score gauge on a canvas element.
 * @param {string} canvasId - Canvas element ID
 * @param {number} score - Score value 0-100
 * @param {string} color - Color for the filled arc
 */
function drawScoreGauge(canvasId, score, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const center = size / 2;
    const radius = (size / 2) - 15;
    const lineWidth = 12;

    // Clear
    ctx.clearRect(0, 0, size, size);

    // Background arc
    ctx.beginPath();
    ctx.arc(center, center, radius, -0.5 * Math.PI, 1.5 * Math.PI);
    ctx.strokeStyle = '#e9ecef';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Score arc
    if (score > 0) {
        const endAngle = (-0.5 + (score / 100) * 2) * Math.PI;
        ctx.beginPath();
        ctx.arc(center, center, radius, -0.5 * Math.PI, endAngle);
        ctx.strokeStyle = color || getScoreColor(score);
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();
    }
}

/**
 * Get color for a score value.
 */
function getScoreColor(score) {
    if (score >= 80) return '#198754';  // green
    if (score >= 60) return '#ffc107';  // yellow
    return '#dc3545';                    // red
}

/**
 * Create a Chart.js line chart for trends.
 * @param {string} canvasId - Canvas element ID
 * @param {Array} data - Array of {timestamp, score} objects
 * @param {string} label - Dataset label
 * @param {string} color - Line color
 */
function createTrendChart(canvasId, data, label, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data || data.length === 0) return null;

    const labels = data.map(d => {
        const date = new Date(d.timestamp + 'Z');
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    });
    const scores = data.map(d => d.score);

    return new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label || 'Score',
                data: scores,
                borderColor: color || '#0d6efd',
                backgroundColor: (color || '#0d6efd') + '20',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: (val) => val + '%'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        maxTicksLimit: 10
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%`
                    }
                }
            }
        }
    });
}

/**
 * Create a horizontal bar chart for rule distribution.
 */
function createRuleDistributionChart(canvasId, ruleCounts) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !ruleCounts || Object.keys(ruleCounts).length === 0) return null;

    const labels = Object.keys(ruleCounts);
    const counts = Object.values(ruleCounts);

    const colors = [
        '#dc3545', '#fd7e14', '#ffc107', '#198754', '#0dcaf0',
        '#0d6efd', '#6610f2', '#d63384', '#6c757d', '#20c997'
    ];

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Occurrences',
                data: counts,
                backgroundColor: labels.map((_, i) => colors[i % colors.length]),
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: { stepSize: 1 }
                }
            }
        }
    });
}

/**
 * Draw multiple small score gauges for Lighthouse results.
 */
function drawLighthouseGauges(scores) {
    if (!scores) return;

    const categories = [
        { key: 'performance', label: 'Performance', canvasId: 'gauge-performance' },
        { key: 'accessibility', label: 'Accessibility', canvasId: 'gauge-accessibility' },
        { key: 'best_practices', label: 'Best Practices', canvasId: 'gauge-best-practices' },
        { key: 'seo', label: 'SEO', canvasId: 'gauge-seo' },
    ];

    categories.forEach(cat => {
        const score = scores[cat.key];
        if (score !== null && score !== undefined) {
            drawScoreGauge(cat.canvasId, score, getScoreColor(score));
        }
    });
}


// =========================================================
// Live Running Log Console
// =========================================================

/**
 * Submit a check form via AJAX to an API endpoint, then show a live
 * console that polls for log entries until the check completes.
 *
 * @param {Object} opts
 * @param {string} opts.apiUrl        - POST endpoint (e.g. "/api/html-validation/run")
 * @param {Object} opts.payload       - JSON body for the POST
 * @param {string} opts.containerId   - ID of the DOM element to render the console into
 * @param {string} opts.redirectBase  - Base path to redirect to after completion (e.g. "/html-validation/")
 */
function runCheckWithLiveLog(opts) {
    const container = document.getElementById(opts.containerId);
    if (!container) return;

    // Build the console UI
    container.innerHTML = `
        <div class="card env-card mb-4" id="live-log-card">
            <div class="card-header bg-dark text-white d-flex align-items-center justify-content-between">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm text-light me-2" id="log-spinner"></div>
                    <h6 class="mb-0"><i class="bi bi-terminal me-2"></i>Running Check...</h6>
                </div>
                <div class="d-flex align-items-center gap-3">
                    <div class="progress" style="width: 120px; height: 8px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated bg-info"
                             id="log-progress" style="width: 0%"></div>
                    </div>
                    <span class="badge bg-info" id="log-status-badge">Running</span>
                </div>
            </div>
            <div class="card-body p-0">
                <div id="log-entries"
                     style="background: #1e1e1e; color: #d4d4d4; font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
                            font-size: 0.8rem; padding: 1rem; max-height: 400px; overflow-y: auto; min-height: 120px;">
                </div>
            </div>
        </div>
    `;

    const logEntries = document.getElementById('log-entries');
    const logProgress = document.getElementById('log-progress');
    const logSpinner = document.getElementById('log-spinner');
    const logStatusBadge = document.getElementById('log-status-badge');
    const logCard = document.getElementById('live-log-card');

    // Color map for log levels
    const levelColors = {
        info:    '#569cd6',
        success: '#4ec9b0',
        warning: '#dcdcaa',
        error:   '#f44747',
        progress:'#c586c0',
    };

    const levelIcons = {
        info:    'bi-info-circle',
        success: 'bi-check-circle-fill',
        warning: 'bi-exclamation-triangle-fill',
        error:   'bi-x-circle-fill',
        progress:'bi-arrow-right-circle',
    };

    function appendLog(message, level) {
        const color = levelColors[level] || '#d4d4d4';
        const icon = levelIcons[level] || 'bi-dot';
        const time = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
        const line = document.createElement('div');
        line.style.cssText = `color: ${color}; padding: 2px 0; line-height: 1.6;`;
        line.innerHTML = `<span style="color:#6a9955">${time}</span> <i class="bi ${icon}" style="font-size:0.75rem"></i> ${escapeHtml(message)}`;
        logEntries.appendChild(line);
        logEntries.scrollTop = logEntries.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    appendLog('Submitting check request...', 'info');

    // Step 1: POST to API
    fetch(opts.apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts.payload),
    })
    .then(resp => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    })
    .then(data => {
        const runId = data.run_id;
        appendLog(`Check started — Run #${runId}`, 'success');

        // Step 2: Poll for log entries
        let sinceIndex = 0;
        let pollInterval = null;

        function pollLogs() {
            fetch(`/api/runs/${runId}/logs?since=${sinceIndex}`)
                .then(r => r.json())
                .then(logData => {
                    // Append new entries
                    if (logData.entries && logData.entries.length > 0) {
                        logData.entries.forEach(entry => {
                            appendLog(entry.message, entry.level);
                        });
                        sinceIndex = logData.total_entries;
                    }

                    // Update progress bar
                    if (logData.progress_pct !== undefined) {
                        logProgress.style.width = logData.progress_pct + '%';
                    }

                    // Check if done
                    if (logData.status === 'completed') {
                        clearInterval(pollInterval);
                        logSpinner.classList.add('d-none');
                        logStatusBadge.textContent = 'Completed';
                        logStatusBadge.classList.remove('bg-info');
                        logStatusBadge.classList.add('bg-success');
                        logProgress.style.width = '100%';
                        logProgress.classList.remove('progress-bar-animated', 'progress-bar-striped');
                        logProgress.classList.replace('bg-info', 'bg-success');
                        appendLog('Check completed. Redirecting to results...', 'success');
                        setTimeout(() => {
                            window.location.href = opts.redirectBase + '?run_id=' + runId;
                        }, 1500);
                    } else if (logData.status === 'failed') {
                        clearInterval(pollInterval);
                        logSpinner.classList.add('d-none');
                        logStatusBadge.textContent = 'Failed';
                        logStatusBadge.classList.remove('bg-info');
                        logStatusBadge.classList.add('bg-danger');
                        logProgress.classList.remove('progress-bar-animated', 'progress-bar-striped');
                        logProgress.classList.replace('bg-info', 'bg-danger');
                        appendLog('Check failed. You can view partial results below.', 'error');
                        // Add a link to results
                        const link = document.createElement('div');
                        link.style.cssText = 'padding: 8px 0;';
                        link.innerHTML = `<a href="${opts.redirectBase}?run_id=${runId}" class="text-info">View run details &rarr;</a>`;
                        logEntries.appendChild(link);
                    }
                })
                .catch(err => {
                    appendLog('Error polling logs: ' + err.message, 'error');
                });
        }

        // Start polling every 1 second
        pollInterval = setInterval(pollLogs, 1000);
        // Also poll immediately
        pollLogs();
    })
    .catch(err => {
        appendLog('Failed to start check: ' + err.message, 'error');
        logSpinner.classList.add('d-none');
        logStatusBadge.textContent = 'Error';
        logStatusBadge.classList.remove('bg-info');
        logStatusBadge.classList.add('bg-danger');
    });
}
