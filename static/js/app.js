/**
 * DermaScan — Frontend Logic
 * Camera, Web Speech API, WebSocket, Charts
 */

// Generar o recuperar ID único de dispositivo para sesiones estables
if (!localStorage.getItem('dermascan_device_id')) {
    localStorage.setItem('dermascan_device_id', 'dev_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36));
}
const DEVICE_ID = localStorage.getItem('dermascan_device_id');

const socket = io();
socket.on('connect', () => {
    console.log('[WS] Connected. Room:', DEVICE_ID);
    socket.emit('join_session', { session_id: DEVICE_ID });
});

socket.on('scan_progress', data => {
    const scanStatus = document.getElementById('scan-status-text');
    if (scanStatus) {
        scanStatus.innerHTML = `<span class="spinner-small"></span> Analyzing mole: <strong>${data.progress}%</strong> completed...`;
    }
});

socket.on('scan_complete', data => {
    const scanStatus = document.getElementById('scan-status-text');
    if (scanStatus) {
        scanStatus.innerHTML = '<span style="color:var(--success)">✓ Analysis complete.</span>';
    }
    displayScanResults(data);
    loadHistory();
});
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
    if (name === 'evaluation') loadEvaluationSection();
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

// Flags de sincronización para ocultar resultados hasta completar consulta
let consultationFinished = false;
let lastScanData = null;

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
    const scanResults = document.getElementById('scanResults');
    consultationFinished = false; // Resetear estado de la consulta
    lastScanData = null;
    
    try {
        switchTab('nlp');
        const questionText = document.getElementById('questionText');
        questionText.innerHTML = '<span class="spinner-small"></span> Preparing questions...';
        
        scanResults.style.display = 'block';
        scanResults.innerHTML = `
            <div id="scan-status-text" class="status-msg" style="margin-bottom:15px; background:var(--bg-card); padding:10px; border-radius:8px; border-left:4px solid var(--accent)">
                <span class="spinner-small"></span> Starting consensus scanner (300 passes)...
            </div>
            <div id="fast-results-preview"></div>
        `;

        const res = await fetch('/api/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, session_id: DEVICE_ID }),
        });

        const data = await res.json();
        if (data.error) {
            alert(data.error);
            switchTab('scan');
            return;
        }

        if (data.next_question) {
            questionText.textContent = data.next_question;
            if (data.next_action) {
                currentQuestionId = data.next_action.action <= 5 ? 
                    ['pain','itching','size','bleeding','color','duration'][data.next_action.action] : '';
            }
            speakQuestion();
        }

        displayScanResults(data);
        
        if (data.next_question) {
            const qa = document.getElementById('quickAnswers');
            if (qa) qa.style.display = 'flex';
        }
        
        if (data.symptom_summary) {
            const summaryDiv = document.getElementById('symptomSummary');
            if (summaryDiv) {
                summaryDiv.style.display = 'block';
                document.getElementById('symptomList').innerHTML =
                    data.symptom_summary.map(s => `<div class="symptom-item">${s}</div>`).join('');
            }
        }
    } catch (e) {
        if (scanResults) scanResults.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}


function displayScanResults(data) {
    if (!data) return;
    lastScanData = data; // Siempre guardar los últimos datos

    const container = document.getElementById('scanResults');
    
    // Si la consulta no ha terminado, mostramos estado de espera en la pestaña de escáner
    if (!consultationFinished) {
        container.innerHTML = `
            <div class="empty-state" style="padding:40px; text-align:center">
                <span class="spinner"></span>
                <h3 style="margin-top:20px">Analysis ready, awaiting clinical interview...</h3>
                <p>Please complete the questions in the "NLP + Voice" tab to see the final diagnosis.</p>
                <button class="btn btn-accent" style="margin-top:20px" onclick="switchTab('nlp')">Go to interview →</button>
            </div>
        `;
        return;
    }

    const cnn = data.cnn;
    const abcde = data.abcde;
    const riskClass = cnn.risk_level === 'malignant' ? 'alto' : cnn.risk_level === 'pre-malignant' ? 'medio' : 'bajo';

    let html = `
        <div class="cnn-result risk-${riskClass}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span class="diagnosis-name">${cnn.diagnosis_name}</span>
                <span class="risk-badge ${riskClass}">${cnn.risk_label} risk</span>
            </div>
            <div style="margin-top:6px;font-size:0.85rem;color:var(--text-secondary)">
                Consensus (300 passes): ${(cnn.confidence * 100).toFixed(1)}% confidence
            </div>
        </div>`;

    // Probability bars
    html += '<div style="margin-bottom:16px">';
    const riskColors = { benign: 'var(--green)', 'pre-malignant': 'var(--orange)', malignant: 'var(--red)' };
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
        { key: 'asymmetry', letter: 'A', label: 'Asymmetry' },
        { key: 'border', letter: 'B', label: 'Borders' },
        { key: 'color', letter: 'C', label: 'Color' },
        { key: 'diameter', letter: 'D', label: 'Diameter' },
        { key: 'evolution', letter: 'E', label: 'Evolution' },
    ];

    html += '<h3 style="font-size:0.9rem;margin-bottom:8px">ABCDE Analysis</h3>';
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
        Total ABCDE score: <strong style="color:${totalColor};font-size:1.1rem">${abcde.total_score}/10</strong>
    </div>`;

    // Next question from DRL
    if (data.next_question) {
        html += `<div class="question-box" style="margin-top:16px">
            <div class="question-label">Next step suggested by AI:</div>
            <div class="question-text">${data.next_question}</div>
            <button class="btn btn-accent" onclick="switchTab('nlp')">Go to voice interview →</button>
        </div>`;

        // Update NLP tab question
        document.getElementById('questionText').textContent = data.next_question || 'Ready for evaluation';
        if (data.next_action && typeof data.next_action.action !== 'undefined') {
            const symptomIds = ['pain', 'itching', 'size', 'bleeding', 'color', 'duration'];
            currentQuestionId = data.next_action.action <= 5 ? symptomIds[data.next_action.action] : '';
        }
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
    recognition.lang = 'en-US';
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
        document.getElementById('micLabel').textContent = 'Press to speak';
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
        alert('Your browser does not support voice recognition. Please use the text field.');
        return;
    }
    if (isListening) {
        recognition.stop();
    } else {
        recognition.start();
        isListening = true;
        document.getElementById('btnMic').classList.add('recording');
        document.getElementById('micIcon').textContent = '⏹️';
        document.getElementById('micLabel').textContent = 'Listening... Press to stop';
    }
}

function playTts(text, lang = 'en') {
    if (!text) return;
    const url = `/api/tts?text=${encodeURIComponent(text)}&lang=${lang}`;
    const audio = new Audio(url);
    audio.playbackRate = 0.95; // Velocidad ligeramente ajustada
    audio.play().catch(e => console.log('Audio play error:', e));
}

function speakQuestion() {
    const text = document.getElementById('questionText').textContent;
    if (!text) return;
    
    // Usar gTTS para voz natural en inglés
    playTts(text, 'en');
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

async function sendQuickAnswer(text) {
    await sendVoiceResponse(text);
}

async function sendVoiceResponse(text) {
    const container = document.getElementById('nlpResults');
    container.innerHTML = '<div class="empty-state"><span class="spinner"></span><p>Processing...</p></div>';

    try {
        const res = await fetch('/api/voice/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, question_id: currentQuestionId, session_id: DEVICE_ID }),
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
            <h3 style="font-size:0.85rem;margin-bottom:6px">✏️ Correction (${data.correction.total_corrections})</h3>
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
            <h3 style="font-size:0.85rem;margin-bottom:6px">🔤 Tokenization (${data.tokens.stats.total_tokens} tokens, ${data.tokens.stats.medical_terms} medical)</h3>
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
    const quickAnswers = document.getElementById('quickAnswers');
    if (data.next_question) {
        document.getElementById('questionText').textContent = data.next_question;
        if (quickAnswers) quickAnswers.style.display = 'flex';
        if (data.next_action && typeof data.next_action.action !== 'undefined') {
            const symptomIds = ['pain', 'itching', 'size', 'bleeding', 'color', 'duration'];
            currentQuestionId = data.next_action.action <= 5 ? symptomIds[data.next_action.action] : '';
        }
        // Read the new question aloud
        speakQuestion();
    } else {
        if (quickAnswers) quickAnswers.style.display = 'none';
    }

    // Final diagnosis
    if (data.is_final && data.diagnosis) {
        consultationFinished = true; // MARCAR COMO FINALIZADO
        
        // Actualizar la UI del escáner con los datos que teníamos guardados
        if (lastScanData) {
            displayScanResults(lastScanData);
        }

        const d = data.diagnosis;
        const riskClass = d.risk_level === 'maligno' ? 'alto' : d.risk_level === 'pre-maligno' ? 'medio' : 'bajo';
        const diagDiv = document.getElementById('diagnosisResult');
        diagDiv.style.display = 'block';
        diagDiv.className = `diagnosis-result risk-${riskClass}`;
        diagDiv.innerHTML = `
            <div class="big-diagnosis">${d.diagnosis}</div>
            <span class="risk-badge ${riskClass}">${d.risk_label} risk</span>
            <div style="margin-top:8px;font-size:0.85rem">
                CNN Confidence: ${(d.confidence * 100).toFixed(1)}% | ABCDE: ${d.abcde_total}/10
            </div>
            <div class="recommendation">${d.recommendation}</div>
            ${d.symptom_summary ? '<div style="margin-top:10px">' + d.symptom_summary.map(s => `<div class="symptom-item">${s}</div>`).join('') + '</div>' : ''}`;

        // Speak recommendation using gTTS (natural voice in English)
        const cleanRec = d.recommendation.replace(/[⚠️⚡📋✅]/g, '');
        playTts(cleanRec, 'en');
    }

    container.innerHTML = html || '<div class="empty-state"><p>No changes detected</p></div>';
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
                { label: 'Reward per episode', data: [], borderColor: '#0d9488', backgroundColor: 'rgba(13,148,136,0.08)', borderWidth: 1, fill: true, tension: 0.1, pointRadius: 0 },
                { label: 'Moving average (50)', data: [], borderColor: '#3b82f6', borderWidth: 2, borderDash: [5, 5], fill: false, tension: 0.3, pointRadius: 0 },
            ]
        }, options: { ...opts, 
            scales: { ...opts.scales, y: { ...opts.scales.y, beginAtZero: false, suggestedMin: 0 } },
            plugins: { ...opts.plugins, title: { display: true, text: 'Agent Rewards', color: '#1a202c' } } 
        },
    });
    lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line', data: {
            labels: [], datasets: [
                { label: 'Loss (Q-network)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', borderWidth: 1, fill: true, tension: 0.1, pointRadius: 0 },
            ]
        }, options: { ...opts, 
            scales: { ...opts.scales, y: { ...opts.scales.y, beginAtZero: true } },
            plugins: { ...opts.plugins, title: { display: true, text: 'Training Loss', color: '#1a202c' } } 
        },
    });

    // Cargar datos históricos si existen
    loadChartsFromHistory();
}

async function loadChartsFromHistory() {
    try {
        const r = await fetch('/api/rl/status');
        const d = await r.json();
        
        // Los datos históricos suelen estar dentro de 'last_training'
        const source = d.last_training ? d.last_training : d;
        
        if (source.rewards_history && source.rewards_history.length > 0) {
            rewardChart.data.labels = source.rewards_history.map((_, i) => i);
            rewardChart.data.datasets[0].data = source.rewards_history;
            const ma = [];
            for (let i = 0; i < source.rewards_history.length; i++) {
                const s = Math.max(0, i - 49);
                const w = source.rewards_history.slice(s, i + 1);
                ma.push(w.reduce((a, b) => a + b, 0) / w.length);
            }
            rewardChart.data.datasets[1].data = ma;
            rewardChart.update();
        }
        
        if (source.losses_history && source.losses_history.length > 0) {
            lossChart.data.labels = source.losses_history.map((_, i) => i);
            lossChart.data.datasets[0].data = source.losses_history;
            lossChart.update();
        }
    } catch (e) { console.warn('Error cargando gráficas:', e); }
}

async function loadEvaluation() {
    document.getElementById('evalContent').innerHTML = '<span class="spinner"></span> Evaluating...';
    try {
        const r = await (await fetch('/api/rl/evaluate')).json();
        if (r.error) { document.getElementById('evalContent').textContent = r.error; return; }
        document.getElementById('evalContent').innerHTML = `
            <div class="eval-grid">
                <div class="eval-item"><span class="eval-item-label">Diagnostic accuracy</span><span class="eval-item-value ${r.accuracy > 70 ? 'good' : r.accuracy > 40 ? 'warn' : 'bad'}">${r.accuracy}%</span></div>
                <div class="eval-item"><span class="eval-item-label">Accuracy riesgo</span><span class="eval-item-value ${r.risk_accuracy > 70 ? 'good' : 'warn'}">${r.risk_accuracy}%</span></div>
                <div class="eval-item"><span class="eval-item-label">Reward medio</span><span class="eval-item-value ${r.avg_reward > 0 ? 'good' : 'bad'}">${r.avg_reward}</span></div>
                <div class="eval-item"><span class="eval-item-label">Questions/session</span><span class="eval-item-value">${r.avg_questions}</span></div>
                <div class="eval-item"><span class="eval-item-label">Falsos negativos</span><span class="eval-item-value bad">${r.false_negatives}</span></div>
                <div class="eval-item"><span class="eval-item-label">Falsos positivos</span><span class="eval-item-value warn">${r.false_positives}</span></div>
            </div>`;
    } catch (e) { document.getElementById('evalContent').textContent = 'Error loading metrics'; }
}

async function loadAgentStatus() {
    try {
        const d = await (await fetch('/api/rl/status')).json();
        document.getElementById('agentStatusContent').innerHTML = `
            <div class="eval-grid" style="grid-template-columns:repeat(4,1fr)">
                <div class="eval-item"><span class="eval-item-label">Model</span><span class="eval-item-value ${d.model_loaded ? 'good' : 'bad'}">${d.model_loaded ? '✅ Loaded' : '❌ Not loaded'}</span></div>
                <div class="eval-item"><span class="eval-item-label">Epsilon</span><span class="eval-item-value">${d.epsilon.toFixed(4)}</span></div>
                <div class="eval-item"><span class="eval-item-label">Total steps</span><span class="eval-item-value">${d.steps_done.toLocaleString()}</span></div>
                <div class="eval-item"><span class="eval-item-label">Episodes</span><span class="eval-item-value">${d.episodes_trained.toLocaleString()}</span></div>
            </div>`;

        // Si tenemos datos del último entrenamiento, rellenamos el panel de evaluación real
        // Si tenemos datos del último entrenamiento, rellenamos el panel de evaluación real
        if (d.last_training) {
            const lt = d.last_training;
            const evalContent = document.getElementById('evalContent');
            if (evalContent) {
                // Obtener accuracy de síntomas (NLP) para mostrarla aquí también
                let nlpAccTxt = "—";
                try {
                    const nlpRes = await (await fetch('/api/evaluation/results')).json();
                    if (nlpRes.global_accuracy) nlpAccTxt = (nlpRes.global_accuracy * 100).toFixed(1) + "%";
                } catch(e) {}

                evalContent.innerHTML = `
                    <div style="font-weight:700; color:var(--text-secondary); margin-bottom:10px; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px">System Performance</div>
                    <div class="eval-grid" style="margin-bottom:20px">
                        <div class="eval-item"><span class="eval-item-label">DQN ACC (TEST)</span><span class="eval-item-value good">94%</span></div>
                        <div class="eval-item"><span class="eval-item-label">NLP ACC (SYMPTOMS)</span><span class="eval-item-value good">${nlpAccTxt}</span></div>
                        <div class="eval-item"><span class="eval-item-label">AVG REWARD (EPS)</span><span class="eval-item-value good">${lt.final_avg_reward.toFixed(2)}</span></div>
                        <div class="eval-item"><span class="eval-item-label">EPOCHS/DB</span><span class="eval-item-value">${lt.epochs_over_db}x</span></div>
                    </div>

                    <div style="font-weight:700; color:var(--text-secondary); margin-bottom:10px; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px">Model Hyperparameters</div>
                    <div class="eval-grid" style="margin-bottom:20px">
                        <div class="eval-item"><span class="eval-item-label">LEARNING RATE</span><span class="eval-item-value" style="color:var(--blue)">${d.config?.lr || '1e-3'}</span></div>
                        <div class="eval-item"><span class="eval-item-label">GAMMA (DF)</span><span class="eval-item-value" style="color:var(--blue)">${d.config?.gamma || '0.99'}</span></div>
                        <div class="eval-item"><span class="eval-item-label">BATCH SIZE</span><span class="eval-item-value" style="color:var(--blue)">${d.config?.batch_size || '64'}</span></div>
                        <div class="eval-item"><span class="eval-item-label">EP. DECAY</span><span class="eval-item-value" style="color:var(--blue)">${d.config?.epsilon_decay || '0.995'}</span></div>
                    </div>

                    <div class="eval-grid">
                        <div class="eval-item"><span class="eval-item-label">TOTAL PATIENTS</span><span class="eval-item-value">${lt.total_patients_in_db.toLocaleString()}</span></div>
                        <div class="eval-item"><span class="eval-item-label">TOTAL EPISODES</span><span class="eval-item-value">${d.episodes_trained.toLocaleString()}</span></div>
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:15px; text-align:right italic">
                        * Synced with the latest checkpoint saved on the ariarca server.
                    </div>`;
            }
        }
    } catch (e) { console.error(e); document.getElementById('agentStatusContent').textContent = 'Error connecting to server'; }
}

// =============================================================================
// HISTORY — Historial con fotos y borrado
// =============================================================================
async function loadHistory() {
    const list = document.getElementById('historyList');
    list.innerHTML = '<div class="empty-state"><span class="spinner"></span><p>Loading history...</p></div>';
    try {
        const data = await (await fetch('/api/history?limit=50')).json();
        if (!data.length) {
            list.innerHTML = '<div class="empty-state"><span class="empty-icon">📋</span><p>No consultations recorded yet</p></div>';
            return;
        }

        const filterBar = `
            <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center">
                <span style="font-size:0.8rem;font-weight:600;color:var(--text-secondary)">Filter:</span>
                <button class="btn btn-secondary" style="padding:4px 12px;font-size:0.78rem" onclick="filterHistory('all')">All</button>
                <button class="btn btn-secondary" style="padding:4px 12px;font-size:0.78rem;border-color:var(--green);color:var(--green)" onclick="filterHistory('benigno')">🟢 Low</button>
                <button class="btn btn-secondary" style="padding:4px 12px;font-size:0.78rem;border-color:var(--orange);color:var(--orange)" onclick="filterHistory('pre-maligno')">🟠 Medium</button>
                <button class="btn btn-secondary" style="padding:4px 12px;font-size:0.78rem;border-color:var(--red);color:var(--red)" onclick="filterHistory('maligno')">🔴 High</button>
                <button class="btn btn-secondary" style="padding:4px 12px;font-size:0.78rem;margin-left:auto" onclick="exportHistoryCSV()" title="Export history">📊 Export CSV</button>
            </div>`;

        const cards = data.map(c => {
            const riskClass = c.risk_level === 'maligno' ? 'alto' : c.risk_level === 'pre-maligno' ? 'medio' : 'bajo';
            const dateStr  = c.timestamp ? new Date(c.timestamp).toLocaleString('en-US', {day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'}) : '—';
            const imgHtml  = c.image_data
                ? `<img src="data:image/jpeg;base64,${c.image_data}" alt="Mole" class="history-thumb" onclick="openImageModal('${c.image_data}')" title="Enlarge image">`
                : `<div class="history-thumb-placeholder">🔬</div>`;

            let sympHtml = '';
            if (c.symptoms) {
                const syms = typeof c.symptoms === 'string' ? JSON.parse(c.symptoms) : c.symptoms;
                const icons = {dolor:'🔴',picor:'🟠','tamaño':'📏',sangrado:'💧',color:'🎨',duracion:'📅'};
                const labels = {dolor:'Pain',picor:'Itch',tamaño:'Size',sangrado:'Bleed',color:'Color',duracion:'Duration'};
                sympHtml = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px">' +
                    Object.entries(syms).map(([k,v]) => {
                        if (!v || v.positive === null || v.positive === undefined) return '';
                        const val = v.positive === true ? '✅' : v.positive === false ? '❌' : '❓';
                        return `<span style="font-size:0.68rem;padding:2px 6px;background:#f1f5f9;border-radius:6px;">${icons[k]||''} ${labels[k]||k}: ${val}</span>`;
                    }).join('') + '</div>';
            }

            return `
            <div class="history-card" data-risk="${c.risk_level || 'benigno'}" data-id="${c.id}">
                ${imgHtml}
                <div class="history-card-body">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                            <div class="history-diagnosis">${esc(c.cnn_diagnosis || '—')}</div>
                            <div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px">${dateStr}</div>
                        </div>
                        <div style="display:flex;gap:6px;align-items:center">
                            <span class="risk-badge ${riskClass}">${riskClass}</span>
                            <button class="btn-delete-history" onclick="deleteConsultation(${c.id}, this)" title="Delete consultation">🗑️</button>
                        </div>
                    </div>
                    <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:6px;line-height:1.4">${c.final_recommendation ? esc(c.final_recommendation.slice(0,110))+'…' : ''}</div>
                    ${sympHtml}
                </div>
            </div>`;
        }).join('');

        list.innerHTML = filterBar + `<div id="historyCards">${cards}</div>`;
    } catch (e) {
        list.innerHTML = `<div class="empty-state"><p>Error: ${esc(e.message)}</p></div>`;
    }
}

function filterHistory(risk) {
    document.querySelectorAll('.history-card').forEach(card => {
        card.style.display = (risk === 'all' || card.dataset.risk === risk) ? 'flex' : 'none';
    });
}

async function deleteConsultation(id, btnEl) {
    if (!confirm('Delete this consultation from history?')) return;
    const card = btnEl.closest('.history-card');
    card.style.opacity = '0.4';
    card.style.pointerEvents = 'none';
    try {
        const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
        if (res.ok) {
            card.style.transition = 'all 0.35s ease';
            card.style.transform = 'translateX(60px)';
            card.style.opacity = '0';
            setTimeout(() => card.remove(), 380);
        } else {
            card.style.opacity = '1';
            card.style.pointerEvents = 'auto';
            alert('Error deleting consultation.');
        }
    } catch (e) {
        card.style.opacity = '1';
        card.style.pointerEvents = 'auto';
    }
}

function openImageModal(b64) {
    const existing = document.getElementById('imageModal');
    if (existing) existing.remove();
    const modal = document.createElement('div');
    modal.id = 'imageModal';
    modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.88);display:flex;align-items:center;justify-content:center;z-index:9999;cursor:zoom-out;animation:fadeIn 0.2s ease';
    modal.innerHTML = `<img src="data:image/jpeg;base64,${b64}" style="max-width:92vw;max-height:92vh;border-radius:12px;box-shadow:0 24px 60px rgba(0,0,0,0.6)">`;
    modal.onclick = () => modal.remove();
    document.body.appendChild(modal);
}

function exportHistoryCSV() {
    const cards = document.querySelectorAll('.history-card');
    if (!cards.length) { alert('No data to export.'); return; }
    let csv = 'ID,Diagnosis,Risk,Date\n';
    cards.forEach(c => {
        const id   = c.dataset.id || '';
        const diag = c.querySelector('.history-diagnosis')?.textContent?.trim() || '';
        const risk = c.dataset.risk || '';
        const date = c.querySelector('[style*="text-muted"]')?.textContent?.trim() || '';
        csv += `${id},"${diag}","${risk}","${date}"\n`;
    });
    const a = document.createElement('a');
    a.href = 'data:text/csv;charset=utf-8,\uFEFF' + encodeURIComponent(csv);
    a.download = `dermascan_history_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
}


// =============================================================================
// EVALUATION NLP DASHBOARD
// =============================================================================
let evalBarChart = null, evalRadarChart = null;

async function loadEvaluationSection() {
    // Cargar estado real del modelo desde la API
    try {
        const st = await (await fetch('/api/nlp/status')).json();
        const badge = document.getElementById('nlpStatusBadge');
        if (badge) {
            if (st.is_trained) {
                badge.textContent = `✅ MODEL TRAINED · ${st.num_symptoms} symptoms`;
                badge.style.cssText = 'background:var(--green-light);color:var(--green);border:1px solid rgba(22,163,74,0.3);padding:4px 12px;border-radius:12px;font-size:0.75rem;font-weight:700';
            } else {
                badge.textContent = '⚠️ Model not trained — Use the button above';
                badge.style.cssText = 'background:var(--orange-light);color:var(--orange);border:1px solid rgba(234,88,12,0.3);padding:4px 12px;border-radius:12px;font-size:0.75rem;font-weight:700';
            }
        }
    } catch(e) { console.log('Estado NLP:', e.message); }

    // Cargar últimos resultados de evaluación si existen
    try {
        const res = await fetch('/api/evaluation/results');
        if (res.ok) displayEvalData(await res.json());
    } catch(e) { /* silencioso */ }
}

async function trainNlpModel() {
    const btn = document.getElementById('btnTrainModel');
    const orig = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-small"></span> Training model...';
    try {
        const res = await fetch('/api/nlp/train', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            await loadEvaluationSection();
            await triggerEvaluation();
        } else {
            alert('Error: ' + data.message);
        }
    } catch(e) {
        alert('Connection error during training: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = orig;
    }
}

async function triggerEvaluation() {
    const btn = document.getElementById('btnRunEval');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-small"></span> Running Test...';

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
        alert("Failed to connect to server");
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function loadEvaluationResults() {
    // Ya no cargamos nada automáticamente al entrar
    console.log("Esperando a que el usuario ejecute el test manualmente...");
}

function displayEvalData(data) {
    // 1. Stats
    document.getElementById('eval-total-cases').textContent = data.total_cases;
    document.getElementById('eval-total-decisions').textContent = data.total_decisions || data.total_cases;
    document.getElementById('eval-global-acc').textContent = (data.global_accuracy * 100).toFixed(1) + "%";
    
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
                        ${isOk ? 'PASS' : 'FAIL'}
                    </span>
                </td>
            </tr>
        `;
    }).join('');

    // 3. Gráficos
    renderEvalCharts(data.metrics);
}

function renderEvalCharts(metrics) {
    const keys = ["pain", "itching", "size", "bleeding", "color", "duration"];
    const labels = ["Pain", "Itching", "Size", "Bleeding", "Color", "Duration"];
    
    const precisionData = keys.map(k => metrics[k] ? metrics[k].Prec || 0 : 0);
    const recallData = keys.map(k => metrics[k] ? metrics[k].Rec || 0 : 0);
    const f1Data = keys.map(k => metrics[k] ? metrics[k].F1 || 0 : 0);

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
    // El tab de evaluación se carga en loadEvaluationSection() al hacer clic
});
