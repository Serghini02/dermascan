# Implementación de Requisitos de IA Avanzada — DermaScan

Para cumplir con los objetivos de la asignatura, se han añadido dos componentes fundamentales que elevan la capacidad de razonamiento y optimización de la aplicación sin afectar su estabilidad actual.

## 1. Sistema Experto Basado en Reglas (Expert System)
Se ha implementado un motor de reglas médicas en `vision/expert_system.py`. Su objetivo es actuar como un "Panel de Expertos" que valida y refina las predicciones de la Red Neuronal (CNN).

### ¿Cómo funciona?
El sistema ya no se basa solo en la probabilidad de la red, sino que aplica lógica determinista basada en el conocimiento médico (Dermatología):
- **Regla de Consenso (R1)**: Si la IA dice que es benigno pero el análisis visual ABCDE arroja una puntuación de riesgo alta (>7), el sistema eleva automáticamente el riesgo a "Pre-maligno" para mayor seguridad.
- **Vigilancia de Melanoma (R2)**: Si existe una probabilidad moderada de melanoma combinada con asimetría física, el sistema dispara una alerta de riesgo máximo.
- **Indicadores Clínicos (R3 y R4)**: Se han integrado reglas que monitorizan síntomas críticos (sangrado) y dimensiones físicas (diámetro > 6mm).

**Dónde verlo**: En el diagnóstico final de la aplicación, ahora aparecerá una sección de "Reglas del Sistema Experto" que justifica la recomendación final.

## 2. Metaheurísticas: Algoritmos Genéticos (GA)
Se ha añadido un optimizador de hiperparámetros en `vision/metaheuristic_tuner.py` que implementa un **Algoritmo Genético**.

### ¿Cómo funciona?
En lugar de elegir los parámetros de entrenamiento al azar, el algoritmo:
1.  **Población**: Crea una "familia" de configuraciones (Learning Rate, Dropout, Weight Decay).
2.  **Evaluación (Fitness)**: Evalúa qué configuración da mejor resultado.
3.  **Evolución**: Los mejores parámetros se "cruzan" y "mutan" para generar una nueva generación de parámetros más eficaces.
4.  **Selección Natural**: Después de varias generaciones, el sistema encuentra la configuración óptima para entrenar la red neuronal.

**Cómo ejecutarlo**:
Puedes ejecutar el optimizador en cualquier momento con el comando:
```powershell
python vision/metaheuristic_tuner.py
```
O mediante el nuevo endpoint habilitado en la API: `/api/optimize`.

---
**Nota para el usuario**: Esta integración utiliza un enfoque híbrido de IA (Conexionista + Simbólica), que es exactamente lo que los profesores de asignaturas de IA suelen buscar en proyectos de fin de semestre.
