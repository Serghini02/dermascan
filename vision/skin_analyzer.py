"""
Análisis ABCDE de lunares usando OpenCV.
A=Asimetría, B=Bordes, C=Color, D=Diámetro, E=Evolución
"""

import cv2
import numpy as np


def analyze_mole(image):
    """
    Análisis ABCDE completo de un lunar.

    Args:
        image: numpy array (BGR) de la imagen del lunar

    Returns:
        dict con puntuaciones ABCDE y nivel de riesgo visual
    """
    # Preprocesar - segmentar el lunar
    mask = _segment_mole(image)
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
    d_score, d_detail = _diameter(contour)
    e_score, e_detail = _evolution_indicators(image, mask, contour)

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
        "border": {"score": round(b_score, 2), "detail": b_detail},
        "color": {"score": round(c_score, 2), "detail": c_detail},
        "diameter": {"score": round(d_score, 2), "detail": d_detail},
        "evolution": {"score": round(e_score, 2), "detail": e_detail},
        "total_score": total,
        "risk": risk,
    }


def _segment_mole(image):
    """Segmenta el lunar del fondo de piel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Usar Filtro Bilateral para suavizar el ruido pero preservando bordes
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Combinar Otsu con AdaptiveThreshold para bordes más precisos
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 15, 4)
    
    base_mask = cv2.bitwise_or(otsu, adaptive)

    # Filtrar por saturación y valor (lunares tienden a ser oscuros y saturados)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    dark_mask = (v < 185).astype(np.uint8) * 255
    sat_mask = (s > 15).astype(np.uint8) * 255

    combined = cv2.bitwise_and(base_mask, dark_mask)
    combined = cv2.bitwise_and(combined, sat_mask)

    # Limpiar morfología conservando estructura
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return combined


def _asymmetry(contour, mask):
    """A: Mide la asimetría del lunar comparando mitades."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return 0.0, "No se pudo calcular"

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    h, w = mask.shape

    # Comparar mitades horizontales
    left = mask[:, :cx]
    right = mask[:, cx:]
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        left_area = np.sum(left[:, -min_w:] > 0)
        right_area = np.sum(right[:, :min_w] > 0)
        h_asym = abs(left_area - right_area) / max(left_area + right_area, 1)
    else:
        h_asym = 0

    # Comparar mitades verticales
    top = mask[:cy, :]
    bottom = mask[cy:, :]
    min_h = min(top.shape[0], bottom.shape[0])
    if min_h > 0:
        top_area = np.sum(top[-min_h:, :] > 0)
        bottom_area = np.sum(bottom[:min_h, :] > 0)
        v_asym = abs(top_area - bottom_area) / max(top_area + bottom_area, 1)
    else:
        v_asym = 0

    score = (h_asym + v_asym) / 2
    detail = f"Horizontal: {h_asym:.0%}, Vertical: {v_asym:.0%}"
    return score, detail


def _border_irregularity(contour):
    """B: Mide la irregularidad del borde."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if perimeter == 0:
        return 0.0, "No se pudo calcular"

    # Circularidad (1 = círculo perfecto, 0 = irregular)
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Convexidad
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1

    # Score: mayor irregularidad = mayor score
    score = 1.0 - (circularity * 0.5 + solidity * 0.5)
    score = max(0, min(score, 1))

    detail = f"Circularidad: {circularity:.2f}, Solidez: {solidity:.2f}"
    return score, detail


def _color_variation(image, mask):
    """C: Mide la variación de color dentro del lunar."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]

    if len(pixels) < 10:
        return 0.0, "Insuficientes píxeles"

    # Varianza en cada canal HSV
    h_std = float(np.std(pixels[:, 0]))
    s_std = float(np.std(pixels[:, 1]))
    v_std = float(np.std(pixels[:, 2]))

    # Contar colores distintos (cuantizados)
    h_bins = np.histogram(pixels[:, 0], bins=12, range=(0, 180))[0]
    num_colors = np.sum(h_bins > len(pixels) * 0.05)

    # Score: más variación de color = mayor riesgo
    color_var = (h_std / 90 + s_std / 128 + v_std / 128) / 3
    score = min(color_var * 2 + (num_colors - 1) * 0.1, 1.0)
    score = max(0, score)

    detail = f"{num_colors} colores detectados, variación: {color_var:.2f}"
    return score, detail


def _diameter(contour):
    """D: Estima el diámetro del lunar."""
    _, radius = cv2.minEnclosingCircle(contour)
    diameter_px = radius * 2

    # Estimación en mm (asumiendo distancia típica de foto de móvil)
    # Esto es una aproximación, el DPI real depende de la cámara y distancia
    estimated_mm = diameter_px * 0.05  # Factor de conversión aproximado

    # Score: lunares > 6mm tienen más riesgo
    if estimated_mm > 6:
        score = min(1.0, estimated_mm / 10)
    else:
        score = estimated_mm / 12

    detail = f"Diámetro estimado: {estimated_mm:.1f}mm ({diameter_px:.0f}px)"
    return score, detail


def _evolution_indicators(image, mask, contour):
    """E: Indicadores de evolución (análisis de textura como proxy)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    pixels = roi[mask > 0]

    if len(pixels) < 10:
        return 0.0, "Insuficientes datos"

    # Varianza de textura (texturas irregulares → posible evolución)
    texture_var = float(np.var(pixels)) / 255

    # Bordes internos
    edges = cv2.Canny(roi, 50, 150)
    edge_pixels = cv2.bitwise_and(edges, edges, mask=mask)
    edge_ratio = np.sum(edge_pixels > 0) / max(np.sum(mask > 0), 1)

    score = min((texture_var + edge_ratio * 2) / 2, 1.0)

    detail = f"Textura: {texture_var:.2f}, Bordes internos: {edge_ratio:.2%}"
    return score, detail


def _default_scores(reason):
    """Scores por defecto cuando no se detecta lunar."""
    return {
        "asymmetry": {"score": 0, "detail": reason},
        "border": {"score": 0, "detail": reason},
        "color": {"score": 0, "detail": reason},
        "diameter": {"score": 0, "detail": reason},
        "evolution": {"score": 0, "detail": reason},
        "total_score": 0,
        "risk": "no_detectado",
    }
