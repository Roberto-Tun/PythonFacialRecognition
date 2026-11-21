"""Configuración de la v2, solo por variables de entorno.

Ninguna llave vive en el código: `V2_LLAVES` llega del entorno (Secret Manager en
Cloud Run, `-e` en `docker run` en local).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet

# La base de KAXTEN en producción. La v2 no debe escribir ahí nunca.
BASE_KAXTEN = "(default)"

MODELO = "facenet_512"

# Mismos pesos (FaceNet 20180402, 512): el teléfono usa la exportación a TFLite
# en fp16 y sus vectores se comparan directamente con los del servidor.
MODELOS_COMPATIBLES: FrozenSet[str] = frozenset({"facenet_512", "facenet_512_fp16"})

DIMENSION = 512


class ConfiguracionInvalida(RuntimeError):
    pass


# El único índice que vive en la base de KAXTEN: sus personas desaparecidas,
# leídas en el formato de la v1 (ver app/v2/kaxten.py).
INDICE_KAXTEN = "desaparecidos"


@dataclass(frozen=True)
class Cliente:
    nombre: str
    # `*` = todos los índices (solo para el modo abierto de pruebas locales).
    indices: FrozenSet[str]
    # `base` para la demo y cualquier cliente nuevo; `(default)` solo para KAXTEN.
    base_datos: str = "base"

    @property
    def es_kaxten(self) -> bool:
        return self.base_datos == BASE_KAXTEN

    def puede(self, indice: str) -> bool:
        return "*" in self.indices or indice in self.indices


@dataclass(frozen=True)
class ConfigV2:
    base_datos: str = "base"
    coleccion: str = "Vectores"
    llaves: Dict[str, Cliente] = field(default_factory=dict)
    sin_llave: bool = False
    refresco_s: float = 30.0
    refresco_kaxten_s: float = 600.0
    limite_max: int = 50
    umbral_defecto: float = 0.5
    max_bytes_foto: int = 8 * 1024 * 1024
    max_fotos: int = 5

    @property
    def modo_autenticacion(self) -> str:
        if self.llaves:
            return "llaves"
        if self.sin_llave:
            return "abierta (solo para pruebas locales)"
        return "sin configurar"


def _leer_llaves(texto: str) -> Dict[str, Cliente]:
    """`V2_LLAVES` es un JSON: {"<llave>": {"cliente": "...", "indices": [...]}}."""
    if not texto.strip():
        return {}
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as e:
        # Sin el texto de la variable en el mensaje: trae llaves.
        raise ConfiguracionInvalida(f"V2_LLAVES no es JSON válido (línea {e.lineno}).") from None
    if not isinstance(datos, dict):
        raise ConfiguracionInvalida("V2_LLAVES debe ser un objeto {llave: {cliente, indices}}.")
    llaves: Dict[str, Cliente] = {}
    for llave, valor in datos.items():
        if not isinstance(valor, dict) or not isinstance(valor.get("indices"), list):
            raise ConfiguracionInvalida("Cada llave de V2_LLAVES necesita 'cliente' e 'indices' (lista).")
        if len(llave) < 24:
            raise ConfiguracionInvalida("Las llaves de V2_LLAVES deben tener al menos 24 caracteres.")
        indices = frozenset(str(i) for i in valor["indices"])
        base = str(valor.get("base_datos") or "base").strip()
        if base == "default":
            base = BASE_KAXTEN
        # La base de KAXTEN solo se abre a una llave que lo declare a propósito,
        # y solo para su índice: un error de configuración no debe dejar que la
        # llave de la demo lea o escriba datos de personas desaparecidas.
        if base == BASE_KAXTEN and indices != {INDICE_KAXTEN}:
            raise ConfiguracionInvalida(
                f"Una llave con base_datos '(default)' solo puede tener el índice '{INDICE_KAXTEN}'."
            )
        if INDICE_KAXTEN in indices and base != BASE_KAXTEN:
            raise ConfiguracionInvalida(
                f"El índice '{INDICE_KAXTEN}' solo existe en la base de KAXTEN: declara base_datos '(default)'."
            )
        if "*" in indices:
            raise ConfiguracionInvalida("'*' no se permite en V2_LLAVES; lista los índices.")
        llaves[llave] = Cliente(
            nombre=str(valor.get("cliente") or "sin_nombre"),
            indices=indices,
            base_datos=base,
        )
    return llaves


def cargar(entorno=None) -> ConfigV2:
    env = os.environ if entorno is None else entorno
    base = env.get("V2_BASE_DATOS", "base").strip()
    if base in ("", BASE_KAXTEN, "default"):
        raise ConfiguracionInvalida(
            "La v2 no puede usar la base (default): es la de KAXTEN en producción. "
            "Usa V2_BASE_DATOS=base."
        )
    return ConfigV2(
        base_datos=base,
        coleccion=env.get("V2_COLECCION", "Vectores").strip() or "Vectores",
        llaves=_leer_llaves(env.get("V2_LLAVES", "")),
        sin_llave=env.get("V2_SIN_LLAVE", "") == "1",
        refresco_s=float(env.get("V2_REFRESCO_S", "30")),
        refresco_kaxten_s=float(env.get("V2_REFRESCO_KAXTEN_S", "600")),
        limite_max=int(env.get("V2_LIMITE_MAX", "50")),
        umbral_defecto=float(env.get("V2_UMBRAL", "0.5")),
    )
