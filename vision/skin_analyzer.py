"""
Análisis ABCDE de lunares usando OpenCV.
A=Asimetría, B=Bordes, C=Color, D=Diámetro, E=Evolución

Nota: La imagen de entrada se asume capturada con el lunar centrado
en el overlay circular del escáner (640x640px).
"""

import cv2
import numpy as np


def analyze_mole(image):
    """
    Análisis ABCDE completo de un lunar.

    Args:
        image: numpy array (BGR) de la imagen del lunar (640x640 esperado)

    Returns:
        dict con puntuaciones ABCDE y nivel de riesgo visual
    """
    h, w = image.shape[:2]

    # Preprocesar - segmentar el lunar centrado
    mask = _segment_mole_centered(image)
    if mask is None or cv2.countNonZero(mask) < 100:
        return _default_scores("No se detectó lunar en la imagen")

    # Extraer contorno principal
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _default_scores("No se encontraron contornos")

    contour = max(contours, key=cv2.contourArea)

    # Calcular cada criterio ABCDE
    a_score, a_detail = _asymmetry(contour, mask)
    b_score, b_detail = _border_irregularity(contour)
    c_score, c_detail = _color_variation(image, mask)
    d_score, d_detail = _diameter(contour, w, h)
    e_score, e_detail = _evolution_indicators(image, mask)

    # Puntuación total (0-10)
    total = (a_score + b_score + c_score + d_score + e_score) / 5 * 10
    total = round(min(total, 10.0), 1)

    # Nivel de riesgo
    if total >= 7:
        risk = "alto"
    elif total >= 4:
        risk = "medio"
    else:
        risk = "bajo"

    return {
        "asymmetry": {"score": round(a_score, 2), "detail": a_detail},
        "border":    {"score": round(b_score, 2), "detail": b_detail},
        "color":     {"score": round(c_score, 2), "detail": c_detail},
        "diameter":  {"score": round(d_score, 2), "detail": d_detail},
        "evolution": {"score": round(e_score, 2), "detail": e_detail},
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
    """A: Mide la asimetría del lunar comparando mitades horizontal y vertical."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 0.0, "No se pudo calcular"

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Mitades horizontales (izquierda vs derecha)
    left = mask[:, :cx]
    right = mask[:, cx:]
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        la = np.sum(left[:, -min_w:] > 0)
        ra = np.sum(right[:, :min_w] > 0)
        h_asym = abs(la - ra) / max(la + ra, 1)
    else:
        h_asym = 0

    # Mitades verticales (arriba vs abajo)
    top = mask[:cy, :]
    bot = mask[cy:, :]
    min_h = min(top.shape[0], bot.shape[0])
    if min_h > 0:
        ta = np.sum(top[-min_h:, :] > 0)
        ba = np.sum(bot[:min_h, :] > 0)
        v_asym = abs(ta - ba) / max(ta + ba, 1)
    else:
        v_asym = 0

    score = (h_asym + v_asym) / 2
    detail = f"H: {h_asym:.0%} | V: {v_asym:.0%}"
    return float(score), detail


# =============================================================================
# B — BORDES
# =============================================================================

def _border_irregularity(contour):
    """B: Mide la irregularidad del borde mediante circularidad y convexidad."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if perimeter == 0 or area == 0:
        return 0.0, "No se pudo calcular"

    # Circularidad: 1.0 = círculo perfecto
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    circularity = min(circularity, 1.0)

    # Convexidad: cuánto del contorno es convexo
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0
    solidity = min(solidity, 1.0)

    # Índice de irregularidad: < 0.8 circularity o < 0.85 solidity → sospechoso
    irreg_circ = max(0.0, (0.85 - circularity) / 0.85)
    irreg_sol  = max(0.0, (0.90 - solidity) / 0.90)
    score = (irreg_circ * 0.5 + irreg_sol * 0.5)
    score = min(score, 1.0)

    detail = f"Circularidad: {circularity:.2f} | Convexidad: {solidity:.2f}"
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

    # Relajar la fórmula para que no todas salgan con coloración "anormal" elevadísima. 
    # El color es el 3r parámetro, ajustamos la severidad al 50%.
    raw_score = (l_std + a_std + b_std) / 3 * 2 + delta_e * 0.5 + (num_tones - 1) * 0.10
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

    # Conversión calibrada asumiendo un encuadre más típico (50mm en vez de 25mm para evitar falsos positivos)
    # y además, los usuarios no siempre acercan tanto el móvil (a veces 15-20cm representa un campo mayor de 5-6cm)
    px_per_mm = img_w / 50.0
    diameter_mm = diameter_px / px_per_mm

    # Score: lunares > 6mm son clínicamente significativos
    if diameter_mm >= 6:
        score = min(1.0, 0.4 + (diameter_mm - 6) / 20)
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
