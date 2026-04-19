"""
Dataset de entrenamiento para el extractor de síntomas.
Basado en los casos de prueba de evaluación.
"""

TRAINING_DATA = [
    # DOLOR
    {"text": "Sí, me duele bastante cuando lo toco.", "labels": {"dolor": 1, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "No me duele para nada, no siento nada.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Tengo algo de molestia cuando presiono la zona.", "labels": {"dolor": 1, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "A veces arde un poco, sobre todo por las noches.", "labels": {"dolor": 1, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "No siento nada especial, está ahí pero sin dolor.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Me duele el lunar.", "labels": {"dolor": 1, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Es doloroso al tacto.", "labels": {"dolor": 1, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Cero dolor.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},

    # PICOR
    {"text": "Sí, me pica mucho, especialmente por la tarde.", "labels": {"dolor": 0, "picor": 1, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "No me pica, nunca he notado picor.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Tengo comezón de vez en cuando, no siempre.", "labels": {"dolor": 0, "picor": 1, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "A veces pica un poco, sí.", "labels": {"dolor": 0, "picor": 1, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Sin picor, todo normal en esa zona.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Me produce mucho picor.", "labels": {"dolor": 0, "picor": 1, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Siento ganas de rascarme.", "labels": {"dolor": 0, "picor": 1, "tamaño": 0, "sangrado": 0, "color": 0}},

    # TAMAÑO
    {"text": "Sí, ha crecido bastante en los últimos meses.", "labels": {"dolor": 0, "picor": 0, "tamaño": 1, "sangrado": 0, "color": 0}},
    {"text": "No, sigue siendo del mismo tamaño que siempre.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Me parece que está más grande que antes.", "labels": {"dolor": 0, "picor": 0, "tamaño": 1, "sangrado": 0, "color": 0}},
    {"text": "Creo que no ha cambiado, lo veo igual que siempre.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Ha aumentado de tamaño notablemente este año.", "labels": {"dolor": 0, "picor": 0, "tamaño": 1, "sangrado": 0, "color": 0}},
    {"text": "Era más pequeño cuando era niño, ahora es más grande.", "labels": {"dolor": 0, "picor": 0, "tamaño": 1, "sangrado": 0, "color": 0}},
    {"text": "Se ha hecho más grande.", "labels": {"dolor": 0, "picor": 0, "tamaño": 1, "sangrado": 0, "color": 0}},
    {"text": "Igual de pequeño.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},

    # SANGRADO
    {"text": "Sí, una vez sangró cuando me lo rasqué.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 1, "color": 0}},
    {"text": "No, nunca ha sangrado en ningún momento.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "A veces sangra un poquito, sin causa aparente.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 1, "color": 0}},
    {"text": "No ningún sangrado, está completamente seco.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Ha sangrado dos veces estas semanas.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 1, "color": 0}},
    {"text": "Me sale sangre de la herida.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 1, "color": 0}},

    # COLOR
    {"text": "Sí, se ha oscurecido bastante, antes era más claro.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 1}},
    {"text": "No ha cambiado de color, sigue igual.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Tiene colores diferentes, como manchas dentro del lunar.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 1}},
    {"text": "El color es el mismo de siempre, uniforme y marrón.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Cambió de color, ahora tiene partes rojizas.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 1}},
    {"text": "Se puso negro de golpe.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 1}},
    {"text": "Colores muy variados.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 1}},

    # MIXTOS
    {"text": "Me pica y también ha cambiado de color, está más oscuro.", "labels": {"dolor": 0, "picor": 1, "tamaño": 0, "sangrado": 0, "color": 1}},
    {"text": "Duele un poco y sangró el otro día.", "labels": {"dolor": 1, "picor": 0, "tamaño": 0, "sangrado": 1, "color": 0}},
    {"text": "No duele, no pica y no ha cambiado de tamaño.", "labels": {"dolor": 0, "picor": 0, "tamaño": 0, "sangrado": 0, "color": 0}},
    {"text": "Creció bastante y me pica.", "labels": {"dolor": 0, "picor": 1, "tamaño": 1, "sangrado": 0, "color": 0}},
    {"text": "No siento dolor ni picor, pero sí noto que ha crecido mucho.", "labels": {"dolor": 0, "picor": 0, "tamaño": 1, "sangrado": 0, "color": 0}},
]
