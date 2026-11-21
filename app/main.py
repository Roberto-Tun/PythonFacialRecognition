import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.v2.config import cargar
from app.v2.errores import ErrorV2
from app.v2.rutas import router as router_v2

# Falla al arrancar, no en la primera petición, si la v2 apunta a la base de
# KAXTEN o trae llaves mal escritas.
cargar()

app = FastAPI(title="API facial", version="2.0.0-a")

# Con V2_SOLO=1 no se monta la v1: el servicio no toca la base de KAXTEN ni
# carga sus modelos al arrancar. Es lo indicado para el servicio de la demo.
if os.environ.get("V2_SOLO") != "1":
    from app.api.routes import router as router_v1

    app.include_router(router_v1)

app.include_router(router_v2)


def _archivos_como_binario(nodo):
    """FastAPI describe los archivos con OpenAPI 3.1 (`contentMediaType`) y la
    página `/docs` (Swagger UI) no lo entiende: en vez del botón para elegir
    archivo pinta cajas de texto, manda texto y la API responde 422. Con
    `format: binary` sí muestra el selector de archivos."""
    if isinstance(nodo, dict):
        if nodo.get("contentMediaType") == "application/octet-stream":
            nodo.pop("contentMediaType")
            nodo["format"] = "binary"
        for v in nodo.values():
            _archivos_como_binario(v)
    elif isinstance(nodo, list):
        for v in nodo:
            _archivos_como_binario(v)


def _openapi():
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi

        esquema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        _archivos_como_binario(esquema)
        app.openapi_schema = esquema
    return app.openapi_schema


app.openapi = _openapi


@app.exception_handler(ErrorV2)
async def _error_v2(_: Request, e: ErrorV2):
    return JSONResponse(status_code=e.estado, content=e.cuerpo())
