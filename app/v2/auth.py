"""Quién llama y a qué índices puede entrar."""

import hmac
from typing import Optional

from app.v2.config import INDICE_KAXTEN, Cliente, ConfigV2
from app.v2.errores import ErrorV2

CLIENTE_LOCAL = Cliente(nombre="local", indices=frozenset({"*"}))


def identificar(config: ConfigV2, llave: Optional[str]) -> Cliente:
    if config.llaves:
        if llave:
            for conocida, cliente in config.llaves.items():
                # Comparación en tiempo constante: no deja adivinar la llave por
                # cuánto tarda en responder.
                if hmac.compare_digest(conocida.encode(), llave.encode()):
                    return cliente
        raise ErrorV2(401, "llave_invalida", "Falta la llave del cliente (encabezado X-Api-Key) o no es válida.")
    if config.sin_llave:
        return CLIENTE_LOCAL
    raise ErrorV2(
        503,
        "sin_configurar",
        "La v2 no tiene llaves configuradas (V2_LLAVES). Para pruebas locales: V2_SIN_LLAVE=1.",
    )


def exigir_indice(cliente: Cliente, indice: str) -> None:
    # Ni el modo abierto de pruebas (`*`) llega a las personas desaparecidas:
    # hace falta la llave de KAXTEN.
    if indice == INDICE_KAXTEN and not cliente.es_kaxten:
        raise ErrorV2(403, "indice_ajeno", "El índice de KAXTEN solo se abre con su llave.")
    if cliente.es_kaxten and indice != INDICE_KAXTEN:
        raise ErrorV2(403, "indice_ajeno", f"La llave de KAXTEN solo abre el índice '{INDICE_KAXTEN}'.")
    if not cliente.puede(indice):
        raise ErrorV2(403, "indice_ajeno", f"Este cliente no tiene acceso al índice '{indice}'.")
