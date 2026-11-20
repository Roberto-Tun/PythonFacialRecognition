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


@app.exception_handler(ErrorV2)
async def _error_v2(_: Request, e: ErrorV2):
    return JSONResponse(status_code=e.estado, content=e.cuerpo())
