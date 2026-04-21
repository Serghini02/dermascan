"""
Análisis ABCDE de lunares usando OpenCV.
A=Asimetría, B=Bordes, C=Color, D=Diámetro, E=Evolución

Nota: La imagen de entrada se asume capturada con el lunar centrado
en el overlay circular del escáner (640x640px).
"""

import cv2
import numpy as np


def analyze_mole(image, n_passes=300, callback=None):
    """
    Análisis ABCDE completo de un lunar.
    Usa un consenso de n_passes pasadas con pequeñas variaciones de color/brillo
    para obtener una media más robusta y estable.

    Args:
        image: numpy array (BGR) de la imagen del lunar (640x640 esperado)
        n_passes: número de pasadas de consenso (por defecto 500)

    Returns:
        dict con puntuaciones ABCDE y nivel de riesgo visual
    """
    h, w = image.shape[:2]

    # Segmentar el lunar (solo una vez, la máscara es estable)
    mask = _segment_mole_centered(image)
    if mask is None or cv2.countNonZero(mask) < 100:
        return _default_scores("No se detectó lunar en la imagen")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _default_scores("No se encontraron contornos")
    contour = max(contours, key=cv2.contourArea)

    # Scores de forma (A, B, D): estos no dependen del color → calcular una vez
    a_score, a_detail = _asymmetry(contour, mask)
    b_score, b_detail = _border_irregularity(contour)
    d_score, d_detail = _diameter(contour, w, h)

    # Scores de color (C, E): consenso de n_passes pasadas con jitter
    c_scores, e_scores = [], []
    rng = np.random.default_rng(42)
    for i in range(n_passes):
        # Pequeña variación de brillo (±5 puntos en escala 0-255)
        alpha = rng.uniform(0.94, 1.06)   # contraste
        beta  = rng.integers(-8, 9)       # brillo
        aug   = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        c, _ = _color_variation(aug, mask)
        e, _ = _evolution_indicators(aug, mask)
        c_scores.append(c)
        e_scores.append(e)
        
        if callback and (i + 1) % 15 == 0:
            # Progreso proporcional (50-100% ya que CNN fue 0-50%)
            p = int(50 + (i / n_passes) * 50)
            callback(p)
            
        # CEDER CONTROL EN CADA PASADA
        import time
        time.sleep(0.001)

    c_score = float(np.mean(c_scores))
    e_score = float(np.mean(e_scores))
    c_detail = f"Media {n_passes} pasadas | Último: {c_scores[-1]:.3f}"
    e_detail = f"Media {n_passes} pasadas | Último: {e_scores[-1]:.3f}"

    # Puntuación total (0-10)
    total = (a_score + b_score + c_score + d_score + e_score) / 5 * 10
    total = round(min(total, 10.0), 1)

    if total >= 7:
        risk = "alto"
    elif total >= 4:
        risk = "medio"
    else:
        risk = "bajo"

    return {
        "asymmetry": {"score": round(a_score, 2), "detail": a_detail},
        "border":    {"score": round(b_score, 2), "detail": b_detail},
        "color":     {"score": round(c_score, 3), "detail": c_detail},
        "diameter":  {"score": round(d_score, 2), "detail": d_detail},
        "evolution": {"score": round(e_score, 3), "detail": e_detail},
        "total_score": total,
        "risk": risk,
    }


# =============================================================================
# SEGMENTACIÓN — centrada (el lunar siempre está en el centro gracias al overlay)
# =============================================================================

def _segment_mole_centered(image):
    """
    Segmenta el lunar asumiendo que ESTÁ en el centro de la imagen
    (garantizado por el overlay circular del escáner).
    Usa una región central de interés + GrabCut simplificado.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # ROI circular central (aprox 60% del frame donde está el lunar)
    roi_r = int(min(w, h) * 0.30)

    # ---- Paso 1: máscara de ROI circular para acotar la búsqueda ----
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), roi_r, 255, -1)

    # ---- DullRazor: Eliminar pelos detectables antes de usar k-means ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_hair = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hair)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_cleaned = cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)

    # ---- Paso 2: Convertir a Lab y aplicar k-means (2 clusters) ----
    lab = cv2.cvtColor(img_cleaned, cv2.COLOR_BGR2Lab)
    pixels_roi = lab[roi_mask > 0].astype(np.float32)

    if len(pixels_roi) < 50:
        return None

    # K-means para separar fondo-piel del lunar
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels_roi, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # El cluster más oscuro (menor L en Lab) es el lunar
    darker_cluster = int(np.argmin(centers[:, 0]))  # canal L
    mole_px_mask = (labels.flatten() == darker_cluster)

    # Reconstruir máscara
    mask = np.zeros((h, w), dtype=np.uint8)
    roi_indices = np.where(roi_mask.flatten() > 0)[0]
    for idx, is_mole in zip(roi_indices, mole_px_mask):
        if is_mole:
            row, col = divmod(idx, w)
            mask[row, col] = 255

    # ---- Paso 3: Limpiar morfología ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # Mantener solo la componente conexa más cercana al centro
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:
        return None

    best = -1
    best_dist = float('inf')
    for i in range(1, num_labels):
        ccx, ccy = centroids[i]
        dist = (ccx - cx) ** 2 + (ccy - cy) ** 2
        if dist < best_dist and stats[i, cv2.CC_STAT_AREA] > 80:
            best_dist = dist
            best = i

    if best == -1:
        return None

    final_mask = np.zeros_like(mask)
    final_mask[label_map == best] = 255
    return final_mask


# =============================================================================
# A — ASIMETRÍA
# =============================================================================

def _asymmetry(contour, mask):
    """
    A: Mide la asimetría del lunar comparando pixel a pixel (XOR) 
    entre la máscara original y sus versiones volteadas.
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0.0, "No se pudo calcular"

    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    
    # Suavizado extra de la máscara para evitar que el ruido de pixelado cuente como asimetría
    mask_smooth = cv2.medianBlur(mask, 7)
    
    # Recortar el lunar centrado en su centroide para evitar errores de alineación
    x, y, w, h = cv2.boundingRect(contour)
    dist_x = max(cx - x, x + w - cx)
    dist_y = max(cy - y, y + h - cy)
    
    y1, y2 = max(0, cy - dist_y), min(mask_smooth.shape[0], cy + dist_y)
    x1, x2 = max(0, cx - dist_x), min(mask_smooth.shape[1], cx + dist_x)
    centered_mask = mask_smooth[y1:y2, x1:x2]
    
    if centered_mask.size == 0:
        return 0.0, "Error en centrado"

    # Simetría Horizontal
    h_flip = cv2.flip(centered_mask, 1)
    if h_flip.shape != centered_mask.shape:
        h_flip = cv2.resize(h_flip, (centered_mask.shape[1], centered_mask.shape[0]))
    h_xor = cv2.bitwise_xor(centered_mask, h_flip)
    h_asym = np.sum(h_xor > 0) / max(np.sum(centered_mask > 0), 1)
    
    # Simetría Vertical
    v_flip = cv2.flip(centered_mask, 0)
    if v_flip.shape != centered_mask.shape:
        v_flip = cv2.resize(v_flip, (centered_mask.shape[1], centered_mask.shape[0]))
    v_xor = cv2.bitwise_xor(centered_mask, v_flip)
    v_asym = np.sum(v_xor > 0) / max(np.sum(centered_mask > 0), 1)
    
    # Ajuste de escala: un lunar normal suele tener ~20% mismatch por su naturaleza orgánica.
    # Restamos este umbral base.
    asym_val = (h_asym + v_asym) / 2
    score = max(0.0, (asym_val - 0.22) / 0.5) 
    score = min(score, 1.0)
    
    detail = f"Asim H: {h_asym:.0%} | V: {v_asym:.0%}"
    return float(score), detail


# =============================================================================
# B — BORDES
# =============================================================================

def _border_irregularity(contour):
    """
    B: Mide la irregularidad del borde mediante el Índice de Distancia Radial (RDV)
    y la Solidez (Convexidad).
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if area == 0 or perimeter == 0:
        return 0.0, "No se pudo calcular"

    # 1. Circularidad (Compactness): 1.0 = círculo perfecto
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # 2. RDV - Radial Distance Variation (Muy sensible a bordes 'serrados')
    M = cv2.moments(contour)
    # Evitar división por cero
    if M["m00"] == 0: return 0.0, "Área cero"
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    distances = []
    for p in contour:
        d = np.sqrt((p[0][0] - cx)**2 + (p[0][1] - cy)**2)
        distances.append(d)
    
    rdv = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
    
    # 3. Solidez (Indentaciones y muescas)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0

    # Score combinado de irregularidad
    # Umbralizado: ignoramos pequeñas variaciones orgánicas
    irreg_rdv = max(0.0, (rdv - 0.08) / 0.15)
    irreg_circ = max(0.0, (0.76 - circularity) / 0.76)
    irreg_sol = max(0.0, (0.94 - solidity) / 0.94)
    
    # Pesos equilibrados: RDV sigue siendo importante pero no dominante
    score = (irreg_rdv * 0.4 + irreg_circ * 0.3 + irreg_sol * 0.3)
    score = min(score * 1.4, 1.0) # Escalado más justo (v12.4)

    detail = f"Serrado (RDV): {rdv:.2f} | Circularidad: {circularity:.2f}"
    return float(score), detail


# =============================================================================
# C — COLOR
# =============================================================================

def _color_variation(image, mask):
    """
    C: Variación de color dentro del lunar.
    Usa espacio CIE Lab para comparación perceptualmente correcta.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab[mask > 0].astype(np.float32)

    if len(pixels) < 20:
        return 0.0, "Insuficientes píxeles"

    # Desviación estándar en cada canal Lab (L=luminosidad, a=rojo-verde, b=amarillo-azul)
    l_std = float(np.std(pixels[:, 0])) / 128.0   # normalizado 0-1
    a_std = float(np.std(pixels[:, 1])) / 128.0
    b_std = float(np.std(pixels[:, 2])) / 128.0

    # ΔE medio: distancia perceptual al color medio
    mean_color = pixels.mean(axis=0)
    delta_e = float(np.mean(np.linalg.norm(pixels - mean_color, axis=1))) / 50.0

    # Contar regiones de color distintas (k-means con k=3)
    num_tones = 1
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, labels_km, _ = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels_km.flatten(), minlength=3)
        num_tones = int(np.sum(counts > len(pixels) * 0.08))  # clusters con >8% de píxeles
    except Exception:
        pass

    # Penalización específica por colores de alerta clínica (Rojo, Azul, Blanco sobre negro)
    has_red = np.any(pixels[:, 1] > 150)
    has_blue = np.any(pixels[:, 2] < 110)
    has_white = np.any(pixels[:, 0] > 200)
    has_black = np.any(pixels[:, 0] < 50)
    
    alert_colors_score = 0.0
    if has_black and (has_red or has_blue or has_white):
        alert_colors_score = 0.5  # Signo claro de posible melanoma
        
    # Relajamos muchísimo la desviación estándar normal para que los marrones suaves 
    # no eleven el score innecesariamente a más de 5.0 o 6.0
    raw_score = (l_std + a_std + b_std) / 4 + delta_e * 0.1 + (num_tones - 1) * 0.05 + alert_colors_score
    score = min(raw_score, 1.0)
    score = max(0.0, score)

    detail = f"{num_tones} tonos | ΔE: {delta_e*50:.1f} | Var L: {l_std*128:.1f}"
    return float(score), detail


# =============================================================================
# D — DIÁMETRO
# =============================================================================

def _diameter(contour, img_w, img_h):
    """
    D: Estima el diámetro del lunar calibrado al tamaño real de la imagen.

    La imagen capturada es 640x640px y el overlay del escáner abarca ~60%
    del ancho visible. Asumimos que el usuario fotografía a ~15cm con un
    móvil típico (≈ 60px/mm a esa distancia para sensor de 12MP, croppeado
    a un cuadrado de 640px que representa ~25mm de ancho real).
    → 1mm ≈ 25.6px  (640px / 25mm)
    """
    _, radius_px = cv2.minEnclosingCircle(contour)
    diameter_px = radius_px * 2

    # Proporción del lunar respecto al frame
    img_area = img_w * img_h
    mole_area_px = cv2.contourArea(contour)
    pct_frame = (mole_area_px / img_area) * 100

    # Puesto que las cámaras móviles pueden enfocar a gran distancia (FOV de 15cm-20cm reales cruzando la pantalla),
    # hacemos la conversión mucho más suave: asumimos que un ancho de pantalla representa unos 150 milímetros.
    px_per_mm = img_w / 150.0
    diameter_mm = diameter_px / px_per_mm

    # Un lunar de 6mm dispararía sospecha.
    if diameter_mm >= 6:
        # A partir de 6mm sube gradualmente (un melanoma avanzado de 15mm daría 10)
        score = min(1.0, 0.4 + (diameter_mm - 6) / 15.0)
    else:
        score = diameter_mm / 15.0

    detail = f"~{diameter_mm:.1f}mm ({diameter_px:.0f}px · {pct_frame:.1f}%)"
    return float(score), detail


# =============================================================================
# E — EVOLUCIÓN (indicadores de textura como proxy)
# =============================================================================

def _evolution_indicators(image, mask):
    """
    E: Proxy de evolución basado en heterogeneidad de textura.
    Mayor textura interna irregular → posible cambio evolutivo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar solo en zona del lunar
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    pixels_vals = roi[mask > 0].astype(np.float32)

    if len(pixels_vals) < 20:
        return 0.0, "Insuficientes datos"

    # 1. Varianza de textura (Laplaciano dentro del lunar)
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    lap_vals = np.abs(laplacian[mask > 0])
    texture_score = float(np.mean(lap_vals)) / 30.0   # normalizado

    # 2. Heterogeneidad de brillo (coef. variación)
    mean_v = float(np.mean(pixels_vals))
    std_v  = float(np.std(pixels_vals))
    cv_brightness = (std_v / mean_v) if mean_v > 0 else 0

    # 3. Bordes internos (Canny dentro del lunar)
    edges = cv2.Canny(roi, 30, 100)
    edge_ratio = np.sum(edges[mask > 0] > 0) / max(np.sum(mask > 0), 1)

    score = min((texture_score * 0.4 + cv_brightness * 0.3 + edge_ratio * 2 * 0.3), 1.0)
    score = max(0.0, score)

    detail = f"Textura: {texture_score*30:.1f} | Var brillo: {cv_brightness:.2f} | Bordes: {edge_ratio:.1%}"
    return float(score), detail


# =============================================================================
# FALLBACK
# =============================================================================

def _default_scores(reason):
    """Scores por defecto cuando no se detecta lunar."""
    return {
        "asymmetry": {"score": 0, "detail": reason},
        "border":    {"score": 0, "detail": reason},
        "color":     {"score": 0, "detail": reason},
        "diameter":  {"score": 0, "detail": reason},
        "evolution": {"score": 0, "detail": reason},
        "total_score": 0,
        "risk": "no_detectado",
    }
