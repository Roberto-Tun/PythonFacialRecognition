"""Datos del sujeto que acompañan a cada coincidencia (nombre, foto…).

La v2 no es dueña de esos datos: los escribe cada app en su colección. Aquí solo
se leen, por id, para que quien busca no tenga que hacer una segunda consulta.
Solo de los sujetos que ya pasaron el umbral (y, en KAXTEN, la visibilidad).
"""

from datetime import date, datetime
from typing import Optional

# Índice → colección de la base `base` donde viven los datos del sujeto.
# Un índice que no está aquí devuelve coincidencias sin `datos`.
COLECCION_POR_INDICE = {
    "miembros": "Miembros",  # socios del check-in (sin foto, a propósito)
    "registros": "Registros",
}


def sanear(valor):
    """Lo que Firestore devuelve, convertido a JSON (fechas, referencias…)."""
    if isinstance(valor, dict):
        return {str(k): sanear(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [sanear(v) for v in valor]
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    path = getattr(valor, "path", None)
    if isinstance(path, str) and hasattr(valor, "id"):  # DocumentReference
        return path
    if hasattr(valor, "latitude") and hasattr(valor, "longitude"):  # GeoPoint
        return {"latitud": valor.latitude, "longitud": valor.longitude}
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    return str(valor)


def foto_de(indice: str, datos: Optional[dict], ruta_vector: str) -> Optional[str]:
    """URL pública de una foto del sujeto, si la hay."""
    if ruta_vector.startswith("http"):
        return ruta_vector  # KAXTEN: la foto con la que se registró esa cara
    if datos and indice == "registros":
        for f in datos.get("fotos") or []:
            if isinstance(f, str) and f.startswith("http"):
                return f
    return None
