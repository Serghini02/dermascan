/**
 * DermaScan — Frontend Logic
 * Camera, Web Speech API, WebSocket, Charts
 */

const socket = io();
socket.on('connect', () => console.log('[WS] Conectado'));
socket.on('training_progress', d => updateTrainingProgress(d));
socket.on('training_complete', d => onTrainingComplete(d));

// =============================================================================
// TABS
// =============================================================================
function switchTab(name) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === `tab-${name}`));
    if (name === 'rl') loadAgentStatus();
    if (name === 'history') loadHistory();
    if (name === 'evaluation') loadEvaluationResults();
}

// =============================================================================
// CAMERA / SCANNER
// =============================================================================
let cameraStream = null;
let edgeDetectionActive = false;
let edgeAnimId = null;
let edgeBuffer = []; // Buffer para suavizado temporal
const BUFFER_SIZE = 3;
let currentQuestionId = ''; // Rastreador de la pregunta activa

async function initCamera() {
    try {
        const video = document.getElementById('cameraVideo');
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 640 } }
        });
        video.srcObject = cameraStream;
        document.getElementById('cameraOverlay').style.display = 'flex';

        // Iniciar detección de bordes cuando el video esté listo
        video.addEventListener('playing', () => {
            startEdgeDetection();
        }, { once: true });
    } catch (e) {
        console.log('Cámara no disponible:', e.message);
    }
}

// =============================================================================
// EDGE DETECTION EN TIEMPO REAL
// =============================================================================

function startEdgeDetection() {
    edgeDetectionActive = true;
    processEdgeFrame();
}

function stopEdgeDetection() {
    edgeDetectionActive = false;
    edgeBuffer = []; // Limpiar buffer
    if (edgeAnimId) {
        cancelAnimationFrame(edgeAnimId);
        edgeAnimId = null;
    }
    // Limpiar canvas
    const ec = document.getElementById('edgeCanvas');
    if (ec) {
        const ctx = ec.getContext('2d');
        ctx.clearRect(0, 0, ec.width, ec.height);
    }
}

function processEdgeFrame() {
    if (!edgeDetectionActive) return;

    const video = document.getElementById('cameraVideo');
    const edgeCanvas = document.getElementById('edgeCanvas');
    if (!video || !edgeCanvas || video.paused || video.ended || video.videoWidth === 0) {
        edgeAnimId = requestAnimationFrame(processEdgeFrame);
        return;
    }

    const w = video.videoWidth;
    const h = video.videoHeight;
    // Usar resolución menor para rendimiento (~160px)
    const scale = Math.min(160 / w, 160 / h);
    const sw = Math.floor(w * scale);
    const sh = Math.floor(h * scale);

    // Ajustar edgeCanvas al tamaño del preview visible
    const previewRect = edgeCanvas.parentElement.getBoundingClientRect();
    edgeCanvas.width = previewRect.width;
    edgeCanvas.height = previewRect.height;

    // Crear canvas temporal para procesar
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = sw;
    tmpCanvas.height = sh;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(video, 0, 0, sw, sh);

    const imgData = tmpCtx.getImageData(0, 0, sw, sh);
    const gray = toGrayscale(imgData.data, sw, sh);
    const blurred = gaussianBlur3x3(gray, sw, sh);
    const currentEdges = sobelEdges(blurred, sw, sh);

    // Suavizado temporal
    edgeBuffer.push(currentEdges);
    if (edgeBuffer.length > BUFFER_SIZE) edgeBuffer.shift();

    const smoothedEdges = new Float32Array(sw * sh);
    for (let i = 0; i < sw * sh; i++) {
        let sum = 0;
        for (let j = 0; j < edgeBuffer.length; j++) sum += edgeBuffer[j][i];
        smoothedEdges[i] = sum / edgeBuffer.length;
    }

    // Dibujar bordes suavizados
    drawEdges(edgeCanvas, smoothedEdges, sw, sh);

    // Throttle a ~15fps para no saturar CPU
    setTimeout(() => {
        edgeAnimId = requestAnimationFrame(processEdgeFrame);
    }, 66);
}

function toGrayscale(data, w, h) {
    const gray = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
        const idx = i * 4;
        gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
    }
    return gray;
}

function gaussianBlur3x3(gray, w, h) {
    const out = new Float32Array(w * h);
    // Kernel 3x3 Gaussiano: [1,2,1; 2,4,2; 1,2,1] / 16
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const i = y * w + x;
            out[i] = (
                gray[(y-1)*w + (x-1)] + 2*gray[(y-1)*w + x] + gray[(y-1)*w + (x+1)] +
                2*gray[y*w + (x-1)]   + 4*gray[y*w + x]     + 2*gray[y*w + (x+1)] +
                gray[(y+1)*w + (x-1)] + 2*gray[(y+1)*w + x] + gray[(y+1)*w + (x+1)]
            ) / 16;
        }
    }
    return out;
}

function sobelEdges(gray, w, h) {
    const mag = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            // Sobel X
            const gx = (
                -gray[(y-1)*w + (x-1)] + gray[(y-1)*w + (x+1)] +
                -2*gray[y*w + (x-1)]   + 2*gray[y*w + (x+1)] +
                -gray[(y+1)*w + (x-1)] + gray[(y+1)*w + (x+1)]
            );
            // Sobel Y
            const gy = (
                -gray[(y-1)*w + (x-1)] - 2*gray[(y-1)*w + x] - gray[(y-1)*w + (x+1)] +
                 gray[(y+1)*w + (x-1)] + 2*gray[(y+1)*w + x] + gray[(y+1)*w + (x+1)]
            );
            mag[y * w + x] = Math.sqrt(gx * gx + gy * gy);
        }
    }
    return mag;
}

function drawEdges(canvas, edges, sw, sh) {
    const ctx = canvas.getContext('2d');
    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    // Obtener posición y radio del círculo guía real del DOM
    const circleEl = document.querySelector('.camera-circle');
    let clipCx = cw / 2, clipCy = ch / 2, clipR = Math.min(cw, ch) * 0.35;
    if (circleEl) {
        const canvasRect = canvas.getBoundingClientRect();
        const circleRect = circleEl.getBoundingClientRect();
        clipCx = circleRect.left - canvasRect.left + circleRect.width / 2;
        clipCy = circleRect.top - canvasRect.top + circleRect.height / 2;
        clipR = circleRect.width / 2;
    }

    // Calcular umbral adaptativo SOLO dentro del círculo (en coordenadas de la imagen pequeña)
    const cx = sw / 2, cy = sh / 2;
    const scaleX = cw / sw;
    const scaleY = ch / sh;
    // Radio del clip en coordenadas de imagen pequeña
    const clipRSmall = clipR / scaleX;

    let sum = 0, count = 0;
    for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
            const dx = (x - cx), dy = (y - cy);
            if (dx * dx + dy * dy < clipRSmall * clipRSmall) {
                sum += edges[y * sw + x];
                count++;
            }
        }
    }
    const mean = count > 0 ? sum / count : 30;
    const threshold = Math.max(mean * 1.8, 25);

    // Aplicar clip circular exacto al canvas (nada fuera del círculo se pintará)
    ctx.save();
    ctx.beginPath();
    ctx.arc(clipCx, clipCy, clipR, 0, Math.PI * 2);
    ctx.clip();

    // Pintar bordes teal solo dentro del clip
    ctx.fillStyle = 'rgba(13, 148, 136, 0.75)';
    for (let y = 1; y < sh - 1; y++) {
        for (let x = 1; x < sw - 1; x++) {
            if (edges[y * sw + x] > threshold) {
                const px = Math.floor(x * scaleX);
                const py = Math.floor(y * scaleY);
                ctx.fillRect(px, py, Math.ceil(scaleX) + 1, Math.ceil(scaleY) + 1);
            }
        }
    }

    ctx.restore();
}

function resetScan() {
    const video = document.getElementById('cameraVideo');
    const img = document.getElementById('capturedImage');
    const container = document.getElementById('scanResults');
    
    video.style.display = 'block';
    img.style.display = 'none';
    img.src = '';
    document.getElementById('cameraOverlay').style.display = 'flex';
    document.getElementById('fileInput').value = '';

    // Reanudar detección de bordes en tiempo real
    startEdgeDetection();

    const btnReset = document.getElementById('btnReset');
    if (btnReset) btnReset.style.display = 'none';
    document.getElementById('btnCapture').style.display = 'inline-flex';

    container.innerHTML = `
        <div class="empty-state">
            <span class="empty-icon">🔬</span>
            <p>Toma una foto de un lunar para analizarlo</p>
        </div>`;

    // Limpiar estado NLP anterior
    document.getElementById('questionText').textContent = "Escanea un lunar primero para iniciar la consulta";
    document.getElementById('btnSpeak').disabled = true;
    document.getElementById('btnMic').disabled = true;
    const nlpRes = document.getElementById('nlpResults');
    if (nlpRes) nlpRes.innerHTML = '<div class="empty-state"><span class="empty-icon">🎤</span><p>Responde a las preguntas por voz o texto</p></div>';
    const symList = document.getElementById('symptomSummary');
    if (symList) symList.style.display = 'none';
    const diagRes = document.getElementById('diagnosisResult');
    if (diagRes) diagRes.style.display = 'none';
}

async function applyZoom(val) {
    document.getElementById('zoomVal').textContent = `${parseFloat(val).toFixed(1)}x`;
    if (!cameraStream) return;
    try {
        const track = cameraStream.getVideoTracks()[0];
        const cap = track.getCapabilities();
        if (cap.zoom) {
            await track.applyConstraints({ advanced: [{ zoom: parseFloat(val) }] });
            return;
        }
    } catch (e) { }

    // Fallback Zoom Digital por software (CSS)
    const video = document.getElementById('cameraVideo');
    if (video) video.style.transform = `scale(${val})`;
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const img = document.getElementById('capturedImage');
    const zoomVal = parseFloat(document.getElementById('zoomSlider')?.value || 1.0);

    const fw = video.videoWidth || 640;
    const fh = video.videoHeight || 640;

    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');

    // Intentar ver si el zoom es hardware o software para calcular el recorte
    let isNativeZoom = false;
    if (cameraStream) {
        try {
            const track = cameraStream.getVideoTracks()[0];
            const cap = track.getCapabilities();
            if (cap.zoom) isNativeZoom = true;
        } catch (e) { }
    }

    if (!isNativeZoom && zoomVal > 1.0) {
        // Zoom por software: Recortar la región central del video
        const size = Math.min(fw, fh) / zoomVal;
        const sx = (fw - size) / 2;
        const sy = (fh - size) / 2;
        ctx.drawImage(video, sx, sy, size, size, 0, 0, 640, 640);
    } else {
        // normal
        ctx.drawImage(video, 0, 0, fw, fh, 0, 0, 640, 640);
    }

    // Parar detección de bordes
    stopEdgeDetection();

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    img.src = dataUrl;
    img.style.display = 'block';
    video.style.display = 'none';
    document.getElementById('cameraOverlay').style.display = 'none';

    const btnReset = document.getElementById('btnReset');
    if (btnReset) btnReset.style.display = 'inline-flex';
    document.getElementById('btnCapture').style.display = 'none';

    sendScan(dataUrl);
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        stopEdgeDetection();
        const img = document.getElementById('capturedImage');
        img.src = ev.target.result;
        img.style.display = 'block';
        document.getElementById('cameraVideo').style.display = 'none';
        document.getElementById('cameraOverlay').style.display = 'none';

        const btnReset = document.getElementById('btnReset');
        if (btnReset) btnReset.style.display = 'inline-flex';
        document.getElementById('btnCapture').style.display = 'none';

        sendScan(ev.target.result);
    };
    reader.readAsDataURL(file);
}

async function sendScan(imageData) {
    const container = document.getElementById('scanResults');
    container.innerHTML = '<div class="empty-state"><span class="spinner"></span><p>Analizando imagen...</p></div>';
    
    const startTime = Date.now();

    try {
        const res = await fetch('/api/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData }),
        });

        // Mientras se procesa (con el delay del servidor), enviamos al usuario a las preguntas
        switchTab('nlp');
        document.getElementById('questionText').innerHTML = '<span class="spinner-small"></span> Ejecutando consenso de 5 modelos... espere';

        const data = await res.json();
        if (data.error) { container.innerHTML = `<p>${data.error}</p>`; return; }

        displayScanResults(data);
        
        // Hablar la primera pregunta automáticamente
        if (data.next_question) {
            setTimeout(() => speakQuestion(), 500);
        }
    } catch (e) {
        container.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}


function displayScanResults(data) {
    const container = document.getElementById('scanResults');
    const cnn = data.cnn;
    const abcde = data.abcde;
    const riskClass = cnn.risk_level === 'maligno' ? 'alto' : cnn.risk_level === 'pre-maligno' ? 'medio' : 'bajo';

    let html = `
        <div class="cnn-result risk-${riskClass}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span class="diagnosis-name">${cnn.diagnosis_name}</span>
                <span class="risk-badge ${riskClass}">${cnn.risk_label} riesgo</span>
            </div>
            <div style="margin-top:6px;font-size:0.85rem;color:var(--text-secondary)">
                Consenso (5 pasadas): ${(cnn.confidence * 100).toFixed(1)}% de precisión
            </div>
        </div>`;

    // Probability bars
    html += '<div style="margin-bottom:16px">';
    const riskColors = { benigno: 'var(--green)', 'pre-maligno': 'var(--orange)', maligno: 'var(--red)' };
    for (const [code, info] of Object.entries(cnn.probabilities)) {
        const pct = (info.probability * 100).toFixed(1);
        const color = riskColors[info.risk] || 'var(--blue)';
        html += `<div class="prob-bar">
            <span class="prob-bar-label">${info.name}</span>
            <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%;background:${color}"></div></div>
            <span class="prob-bar-value">${pct}%</span>
        </div>`;
    }
    html += '</div>';

    // ABCDE Analysis
    const abcdeLetters = [
        { key: 'asymmetry', letter: 'A', label: 'Asimetría' },
        { key: 'border', letter: 'B', label: 'Bordes' },
        { key: 'color', letter: 'C', label: 'Color' },
        { key: 'diameter', letter: 'D', label: 'Diámetro' },
        { key: 'evolution', letter: 'E', label: 'Evolución' },
    ];

    html += '<h3 style="font-size:0.9rem;margin-bottom:8px">Análisis ABCDE</h3>';
    html += '<div class="abcde-grid">';
    for (const item of abcdeLetters) {
        const score = abcde[item.key]?.score || 0;
        const color = score > 0.6 ? 'var(--red)' : score > 0.3 ? 'var(--orange)' : 'var(--green)';
        html += `<div class="abcde-item">
            <div class="abcde-letter" style="color:${color}">${item.letter}</div>
            <div class="abcde-score" style="color:${color}">${(score * 10).toFixed(1)}</div>
            <div class="abcde-label">${item.label}</div>
        </div>`;
    }
    html += '</div>';

    // Total ABCDE
    const totalColor = abcde.total_score > 6 ? 'var(--red)' : abcde.total_score > 3 ? 'var(--orange)' : 'var(--green)';
    html += `<div style="text-align:center;margin-top:10px;font-size:0.85rem">
        Puntuación ABCDE total: <strong style="color:${totalColor};font-size:1.1rem">${abcde.total_score}/10</strong>
    </div>`;

    // Next question from DRL
    if (data.next_question) {
        html += `<div class="question-box" style="margin-top:16px">
            <div class="question-label">Siguiente paso sugerido por IA:</div>
            <div class="question-text">${data.next_question}</div>
            <button class="btn btn-accent" onclick="switchTab('nlp')">Ir a consulta por voz →</button>
        </div>`;

        // Update NLP tab question
        document.getElementById('questionText').textContent = data.next_question;
        currentQuestionId = data.next_action.action <= 5 ? 
            ['dolor','picor','tamaño','sangrado','color','duracion'][data.next_action.action] : '';
        document.getElementById('btnSpeak').disabled = false;
        document.getElementById('btnMic').disabled = false;
    }

    container.innerHTML = html;
}

// =============================================================================
// VOICE / SPEECH
// =============================================================================
let recognition = null;
let isListening = false;

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.log('SpeechRecognition no disponible');
        return;
    }
    recognition = new SpeechRecognition();
    recognition.lang = 'es-ES';
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onresult = (e) => {
        const transcript = Array.from(e.results).map(r => r[0].transcript).join('');
        document.getElementById('transcriptionText').textContent = transcript;
        document.getElementById('transcriptionArea').style.display = 'block';
    };
    recognition.onend = () => {
        isListening = false;
        document.getElementById('btnMic').classList.remove('recording');
        document.getElementById('micIcon').textContent = '🎙️';
        document.getElementById('micLabel').textContent = 'Pulsa para hablar';
    };
    recognition.onerror = (e) => {
        console.log('Speech error:', e.error);
        isListening = false;
        document.getElementById('btnMic').classList.remove('recording');
    };
}

function toggleListening() {
    if (!recognition) initSpeechRecognition();
    if (!recognition) {
        alert('Tu navegador no soporta reconocimiento de voz. Usa el campo de texto.');
        return;
    }
    if (isListening) {
        recognition.stop();
    } else {
        recognition.start();
        isListening = true;
        document.getElementById('btnMic').classList.add('recording');
        document.getElementById('micIcon').textContent = '⏹️';
        document.getElementById('micLabel').textContent = 'Escuchando... Pulsa para parar';
    }
}

function speakQuestion() {
    const text = document.getElementById('questionText').textContent;
    if (!text || !window.speechSynthesis) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-ES';
    utterance.rate = 0.9;
    speechSynthesis.speak(utterance);
}

async function processResponse() {
    const text = document.getElementById('transcriptionText').textContent;
    if (!text) return;
    await sendVoiceResponse(text);
}

async function submitManualResponse() {
    const text = document.getElementById('manualInput').value.trim();
    if (!text) return;
    await sendVoiceResponse(text);
    document.getElementById('manualInput').value = '';
}

async function sendVoiceResponse(text) {
    const container = document.getElementById('nlpResults');
    container.innerHTML = '<div class="empty-state"><span class="spinner"></span><p>Procesando...</p></div>';

    try {
        const res = await fetch('/api/voice/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, question_id: currentQuestionId }),
        });
        const data = await res.json();
        displayNlpResults(data);
    } catch (e) {
        container.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

function displayNlpResults(data) {
    const container = document.getElementById('nlpResults');
    let html = '';

    // Correction
    if (data.correction && data.correction.total_corrections > 0) {
        html += `<div style="margin-bottom:16px">
            <h3 style="font-size:0.85rem;margin-bottom:6px">✏️ Corrección (${data.correction.total_corrections})</h3>
            <div style="font-size:0.85rem;padding:10px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;color:#166534">${data.correction.corrected}</div>`;
        data.correction.corrections.forEach(c => {
            if (c.original && c.original !== c.corrected) {
                html += `<span class="correction-original">${esc(c.original)}</span>
                         <span class="correction-fixed">${esc(c.corrected || '∅')}</span> `;
            }
        });
        html += '</div>';
    }

    // Tokens
    if (data.tokens) {
        html += `<div style="margin-bottom:16px">
            <h3 style="font-size:0.85rem;margin-bottom:6px">🔤 Tokenización (${data.tokens.stats.total_tokens} tokens, ${data.tokens.stats.medical_terms} médicos)</h3>
            <div class="token-container">`;
        data.tokens.token_details.forEach(t => {
            const cls = t.is_medical ? 'medical' : t.is_stopword ? 'stopword' : 'normal';
            html += `<span class="token ${cls}">${esc(t.token)}</span>`;
        });
        html += '</div></div>';
    }

    // Symptoms
    if (data.symptom_summary) {
        const summaryDiv = document.getElementById('symptomSummary');
        summaryDiv.style.display = 'block';
        document.getElementById('symptomList').innerHTML =
            data.symptom_summary.map(s => `<div class="symptom-item">${s}</div>`).join('');
    }

    // Next question
    if (data.next_question) {
        document.getElementById('questionText').textContent = data.next_question;
        if (data.next_action) {
            currentQuestionId = data.next_action.action <= 5 ? 
                ['dolor','picor','tamaño','sangrado','color','duracion'][data.next_action.action] : '';
        }
    }

    // Final diagnosis
    if (data.is_final && data.diagnosis) {
        const d = data.diagnosis;
        const riskClass = d.risk_level === 'maligno' ? 'alto' : d.risk_level === 'pre-maligno' ? 'medio' : 'bajo';
        const diagDiv = document.getElementById('diagnosisResult');
        diagDiv.style.display = 'block';
        diagDiv.className = `diagnosis-result risk-${riskClass}`;
        diagDiv.innerHTML = `
            <div class="big-diagnosis">${d.diagnosis}</div>
            <span class="risk-badge ${riskClass}">${d.risk_label} riesgo</span>
            <div style="margin-top:8px;font-size:0.85rem">
                Confianza CNN: ${(d.confidence * 100).toFixed(1)}% | ABCDE: ${d.abcde_total}/10
            </div>
            <div class="recommendation">${d.recommendation}</div>
            ${d.symptom_summary ? '<div style="margin-top:10px">' + d.symptom_summary.map(s => `<div class="symptom-item">${s}</div>`).join('') + '</div>' : ''}`;

        // Speak recommendation
        if (window.speechSynthesis) {
            const msg = new SpeechSynthesisUtterance(d.recommendation.replace(/[⚠️⚡📋✅]/g, ''));
            msg.lang = 'es-ES'; msg.rate = 0.85;
            speechSynthesis.speak(msg);
        }
    }

    container.innerHTML = html || '<div class="empty-state"><p>Sin cambios detectados</p></div>';
}

// =============================================================================
// DRL — Solo estadísticas y métricas (sin entrenamiento)
// =============================================================================
let rewardChart = null, lossChart = null;

function initCharts() {
    const opts = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#4a5568', font: { family: 'Inter' } } } },
        scales: {
            x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(0,0,0,0.05)' } },
            y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(0,0,0,0.05)' } },
        },
    };
    rewardChart = new Chart(document.getElementById('rewardChart'), {
        type: 'line', data: {
            labels: [], datasets: [
                { label: 'Reward por episodio', data: [], borderColor: '#0d9488', backgroundColor: 'rgba(13,148,136,0.08)', borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0 },
                { label: 'Media móvil (50)', data: [], borderColor: '#3b82f6', borderWidth: 2, borderDash: [5, 5], fill: false, tension: 0.3, pointRadius: 0 },
            ]
        }, options: { ...opts, plugins: { ...opts.plugins, title: { display: true, text: 'Recompensas del Agente', color: '#1a202c' } } },
    });
    lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line', data: {
            labels: [], datasets: [
                { label: 'Loss (Q-network)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0 },
            ]
        }, options: { ...opts, plugins: { ...opts.plugins, title: { display: true, text: 'Pérdida del Entrenamiento', color: '#1a202c' } } },
    });

    // Cargar datos históricos si existen
    loadChartsFromHistory();
}

async function loadChartsFromHistory() {
    try {
        const r = await fetch('/api/rl/status');
        const d = await r.json();
        if (d.rewards_history && d.rewards_history.length > 0) {
            rewardChart.data.labels = d.rewards_history.map((_, i) => i);
            rewardChart.data.datasets[0].data = d.rewards_history;
            const ma = [];
            for (let i = 0; i < d.rewards_history.length; i++) {
                const s = Math.max(0, i - 49);
                const w = d.rewards_history.slice(s, i + 1);
                ma.push(w.reduce((a, b) => a + b, 0) / w.length);
            }
            rewardChart.data.datasets[1].data = ma;
            rewardChart.update();
        }
        if (d.losses_history && d.losses_history.length > 0) {
            lossChart.data.labels = d.losses_history.map((_, i) => i);
            lossChart.data.datasets[0].data = d.losses_history;
            lossChart.update();
        }
    } catch (e) { /* silencioso */ }
}

async function loadEvaluation() {
    document.getElementById('evalContent').innerHTML = '<span class="spinner"></span> Evaluando...';
    try {
        const r = await (await fetch('/api/rl/evaluate')).json();
        if (r.error) { document.getElementById('evalContent').textContent = r.error; return; }
        document.getElementById('evalContent').innerHTML = `
            <div class="eval-grid">
                <div class="eval-item"><span class="eval-item-label">Accuracy diagnóstico</span><span class="eval-item-value ${r.accuracy > 70 ? 'good' : r.accuracy > 40 ? 'warn' : 'bad'}">${r.accuracy}%</span></div>
                <div class="eval-item"><span class="eval-item-label">Accuracy riesgo</span><span class="eval-item-value ${r.risk_accuracy > 70 ? 'good' : 'warn'}">${r.risk_accuracy}%</span></div>
                <div class="eval-item"><span class="eval-item-label">Reward medio</span><span class="eval-item-value ${r.avg_reward > 0 ? 'good' : 'bad'}">${r.avg_reward}</span></div>
                <div class="eval-item"><span class="eval-item-label">Preguntas/sesión</span><span class="eval-item-value">${r.avg_questions}</span></div>
                <div class="eval-item"><span class="eval-item-label">Falsos negativos</span><span class="eval-item-value bad">${r.false_negatives}</span></div>
                <div class="eval-item"><span class="eval-item-label">Falsos positivos</span><span class="eval-item-value warn">${r.false_positives}</span></div>
            </div>`;
    } catch (e) { document.getElementById('evalContent').textContent = 'Error al cargar métricas'; }
}

async function loadAgentStatus() {
    try {
        const d = await (await fetch('/api/rl/status')).json();
        document.getElementById('agentStatusContent').innerHTML = `
            <div class="eval-grid" style="grid-template-columns:repeat(4,1fr)">
                <div class="eval-item"><span class="eval-item-label">Modelo</span><span class="eval-item-value ${d.model_loaded ? 'good' : 'bad'}">${d.model_loaded ? '✅ Cargado' : '❌ No cargado'}</span></div>
                <div class="eval-item"><span class="eval-item-label">Epsilon</span><span class="eval-item-value">${d.epsilon.toFixed(4)}</span></div>
                <div class="eval-item"><span class="eval-item-label">Steps totales</span><span class="eval-item-value">${d.steps_done.toLocaleString()}</span></div>
                <div class="eval-item"><span class="eval-item-label">Episodios</span><span class="eval-item-value">${d.episodes_trained.toLocaleString()}</span></div>
            </div>`;
    } catch (e) { document.getElementById('agentStatusContent').textContent = 'Error al conectar con el servidor'; }
}

// =============================================================================
// HISTORY
// =============================================================================
async function loadHistory() {
    try {
        const data = await (await fetch('/api/history')).json();
        const list = document.getElementById('historyList');
        if (!data.length) { list.innerHTML = '<div class="empty-state"><p>Sin consultas</p></div>'; return; }
        list.innerHTML = data.map(c => {
            const riskClass = c.risk_level === 'maligno' ? 'alto' : c.risk_level === 'pre-maligno' ? 'medio' : 'bajo';
            return `<div class="history-item">
                <div>
                    <span class="history-diagnosis">${c.cnn_diagnosis || '—'}</span>
                    <span class="risk-badge ${riskClass}">${riskClass}</span>
                </div>
                <span class="history-date">${c.timestamp?.split('T')[0] || '—'}</span>
            </div>`;
        }).join('');
    } catch (e) { document.getElementById('historyList').innerHTML = `<p>${e.message}</p>`; }
}

// =============================================================================
// EVALUATION NLP DASHBOARD
// =============================================================================
let evalBarChart = null, evalRadarChart = null;

async function triggerEvaluation() {
    const btn = document.getElementById('btnRunEval');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-small"></span> Ejecutando Test...';

    try {
        const res = await fetch('/api/evaluation/run', { method: 'POST' });
        const data = await res.json();
        
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            // Actualizar la vista con los nuevos datos
            displayEvalData(data);
        }
    } catch (e) {
        console.error(e);
        alert("Fallo en la conexión con el servidor");
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function loadEvaluationResults() {
    try {
        const res = await fetch('/api/evaluation/results');
        const data = await res.json();
        if (data.error) return; // No hay resultados previos
        displayEvalData(data);
    } catch (e) {
        console.log("No se pudieron cargar resultados previos.");
    }
}

function displayEvalData(data) {
    // 1. Stats
    document.getElementById('eval-total-cases').textContent = data.total_cases;
    
    // 2. Tabla de casos
    const tbody = document.getElementById('evalTableBody');
    tbody.innerHTML = data.cases.map(c => {
        const isOk = c.case_accuracy === 1.0;
        return `
            <tr style="border-bottom: 1px solid var(--border-light)">
                <td style="padding: 12px; font-weight: 700">#${c.id}</td>
                <td style="padding: 12px; color: var(--secondary)">${c.question.toUpperCase()}</td>
                <td style="padding: 12px; font-style: italic; font-size: 0.85rem">"${c.text}"</td>
                <td style="padding: 12px">
                    <span class="badge ${isOk ? 'badge-success' : 'badge-danger'}">
                        ${isOk ? 'ACIERTO' : 'FALLO'}
                    </span>
                </td>
            </tr>
        `;
    }).join('');

    // 3. Gráficos
    renderEvalCharts(data.metrics);
}

function renderEvalCharts(metrics) {
    const keys = ["dolor", "picor", "tamaño", "sangrado", "color", "duracion"];
    const labels = ["Dolor", "Picor", "Tamaño", "Sangrado", "Color", "Duración"];
    
    const precisionData = keys.map(k => metrics[k].precision);
    const recallData = keys.map(k => metrics[k].recall);
    const f1Data = keys.map(k => metrics[k].f1);

    // BARRAS
    if (evalBarChart) evalBarChart.destroy();
    evalBarChart = new Chart(document.getElementById('evalBarChart'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Precision', data: precisionData, backgroundColor: '#38bdf8' },
                { label: 'Recall', data: recallData, backgroundColor: '#818cf8' }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                y: { beginAtZero: true, max: 1.2, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            }
        }
    });

    // RADAR
    if (evalRadarChart) evalRadarChart.destroy();
    evalRadarChart = new Chart(document.getElementById('evalRadarChart'), {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'F1 Score',
                data: f1Data,
                backgroundColor: 'rgba(56, 189, 248, 0.2)',
                borderColor: '#38bdf8',
                borderWidth: 2,
                pointBackgroundColor: '#38bdf8'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.05)' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    pointLabels: { color: '#94a3b8', font: { size: 10 } },
                    ticks: { display: false },
                    suggestedMin: 0, suggestedMax: 1
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}


// =============================================================================
// UTILS
// =============================================================================
function esc(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

// =============================================================================
// INIT
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initCamera();
    initSpeechRecognition();
    initCharts();
    loadAgentStatus();
    loadEvaluation();
});



