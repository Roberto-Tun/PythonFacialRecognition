"""Endpoints de la v2. Referencia para quien la consume: docs/v2/API_V2.md."""

import re
import time
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile

from app.v2 import VERSION
from app.v2.auth import exigir_indice, identificar
from app.v2.config import DIMENSION, MODELO, MODELOS_COMPATIBLES, Cliente, ConfigV2, cargar
from app.v2.errores import ErrorV2
from app.v2.esquemas import (
    AltaPorVectores,
    AltaSalida,
    BusquedaPorVector,
    CoincidenciaSalida,
    ResultadoBusquedaSalida,
)
from app.v2.indice import Indices, vector_valido
from app.v2.repositorio import Repositorio, Vector, ahora_iso, id_servidor
from app.v2.vision import MotorVision

router = APIRouter(prefix="/v2", tags=["v2"])

_NOMBRE_INDICE = re.compile(r"^[a-z0-9_]{1,40}$")
# Ids de Firestore generados por la app (20 caracteres) o del dominio del cliente.
_ID_SUJETO = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


# --- Dependencias (las pruebas las reemplazan con dependency_overrides) --------


@lru_cache
def obtener_config() -> ConfigV2:
    return cargar()


@lru_cache
def obtener_repo() -> Repositorio:
    from firebase_admin import firestore

    from app.core.credenciales import inicializar_firebase
    from app.v2.repositorio import RepositorioFirestore

    cfg = obtener_config()
    try:
        app = inicializar_firebase()
        cliente = firestore.client(app=app, database_id=cfg.base_datos)
    except Exception as e:  # DefaultCredentialsError, archivo ilegible…
        # El tipo sí, el texto no: puede traer rutas de la máquina.
        raise ErrorV2(
            503,
            "sin_credenciales",
            f"El servidor no tiene credenciales de Google ({type(e).__name__}). "
            "En local: monta la llave y define GOOGLE_APPLICATION_CREDENTIALS (docs/v2/PRUEBAS_LOCAL.md).",
        ) from None
    return RepositorioFirestore(cliente, cfg.coleccion)


@lru_cache
def obtener_indices() -> Indices:
    return Indices(obtener_repo(), obtener_config().refresco_s)


@lru_cache
def obtener_vision() -> MotorVision:
    from app.v2.vision import MotorFaceNet

    return MotorFaceNet()


def cliente_actual(
    x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key"),
    config: ConfigV2 = Depends(obtener_config),
) -> Cliente:
    return identificar(config, x_api_key)


# --- Validaciones ---------------------------------------------------------------


def _validar_nombres(indice: str, sujeto_id: Optional[str] = None) -> None:
    if not _NOMBRE_INDICE.match(indice):
        raise ErrorV2(400, "indice_invalido", "El índice usa solo minúsculas, números y guion bajo (máx. 40).")
    if sujeto_id is not None and not _ID_SUJETO.match(sujeto_id):
        raise ErrorV2(400, "sujeto_invalido", "El id del sujeto usa letras, números, '_' o '-' (máx. 128).")


def _validar_vectores(modelo: str, vectores: List[List[float]]) -> None:
    if modelo not in MODELOS_COMPATIBLES:
        raise ErrorV2(
            409,
            "modelo_incompatible",
            f"El modelo '{modelo}' no es comparable con el del servidor ({MODELO}). "
            f"Compatibles: {', '.join(sorted(MODELOS_COMPATIBLES))}.",
        )
    for i, v in enumerate(vectores):
        if not vector_valido(v):
            raise ErrorV2(400, "vector_invalido", f"El vector {i} debe tener {DIMENSION} números finitos, no todos cero.")


def _limites(config: ConfigV2, limite: Optional[int], umbral: Optional[float]):
    lim = config.limite_max if limite is None else max(1, min(int(limite), config.limite_max))
    umb = config.umbral_defecto if umbral is None else float(umbral)
    if not 0.0 <= umb <= 1.0:
        raise ErrorV2(400, "umbral_invalido", "El umbral va de 0 a 1.")
    return lim, umb


async def _leer_fotos(fotos: List[UploadFile], config: ConfigV2) -> List[bytes]:
    if not fotos:
        raise ErrorV2(400, "sin_fotos", "Envía al menos una foto en el campo 'fotos'.")
    if len(fotos) > config.max_fotos:
        raise ErrorV2(400, "demasiadas_fotos", f"Máximo {config.max_fotos} fotos por petición.")
    contenidos = []
    for f in fotos:
        datos = await f.read()
        if len(datos) > config.max_bytes_foto:
            raise ErrorV2(413, "foto_grande", f"Cada foto puede pesar hasta {config.max_bytes_foto // (1024 * 1024)} MB.")
        contenidos.append(datos)
    return contenidos


def _ms(desde: float) -> float:
    return round((time.perf_counter() - desde) * 1000, 1)


# --- Rutas ------------------------------------------------------------------------


@router.get("/salud")
def salud(config: ConfigV2 = Depends(obtener_config)):
    """Sin llave: es lo que consultan Cloud Run y la app para saber si hay servicio."""
    return {
        "estado": "ok",
        "version": VERSION,
        "modelo": MODELO,
        "base_datos": config.base_datos,
        "autenticacion": config.modo_autenticacion,
    }


@router.get("/modelo")
def modelo():
    return {
        "modelo": MODELO,
        "compatibles": sorted(MODELOS_COMPATIBLES),
        "dimension": DIMENSION,
        "recorte": "MTCNN, cara más grande, 160×160",
        "normalizacion": "keras_facenet (FaceNet 20180402-114759)",
        "comparacion": "coseno",
    }


@router.post("/vectores")
async def foto_a_vector(
    fotos: List[UploadFile] = File(...),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    vision: MotorVision = Depends(obtener_vision),
):
    """Foto → vector, sin guardar nada. Útil para calibrar o para clientes que
    guardan los vectores por su cuenta."""
    salida = []
    for i, datos in enumerate(await _leer_fotos(fotos, config)):
        cara = vision.cara_principal(datos)
        if cara is None:
            salida.append({"foto": i, "error": "sin_rostro"})
            continue
        salida.append(
            {
                "foto": i,
                "modelo": MODELO,
                "vector": cara.vector,
                "caja": list(cara.caja),
                "confianza": cara.confianza,
                "caras_detectadas": cara.caras_detectadas,
            }
        )
    return {"resultados": salida}


def _guardar(indices: Indices, repo: Repositorio, indice: str, sujeto_id: str, grupo: str, vectores, modelo: str):
    fecha = ahora_iso()
    docs = [
        Vector(
            id=id_servidor(indice, sujeto_id, v),
            indice=indice,
            sujeto_id=sujeto_id,
            grupo=grupo,
            vector=[float(x) for x in v],
            modelo=modelo,
            fecha=fecha,
        )
        for v in vectores
    ]
    repo.guardar(docs)
    indices.invalidar(indice)
    # Sin repetir: la misma cara enviada dos veces es un solo documento.
    return list(dict.fromkeys(d.id for d in docs))


@router.post("/indices/{indice}/sujetos/{sujeto_id}/vectores", response_model=AltaSalida)
def alta_por_vectores(
    indice: str,
    sujeto_id: str,
    cuerpo: AltaPorVectores,
    cliente: Cliente = Depends(cliente_actual),
    repo: Repositorio = Depends(obtener_repo),
    indices: Indices = Depends(obtener_indices),
):
    """Lo que usa la app: el teléfono ya calculó los vectores."""
    _validar_nombres(indice, sujeto_id)
    exigir_indice(cliente, indice)
    _validar_vectores(cuerpo.modelo, cuerpo.vectores)
    ids = _guardar(indices, repo, indice, sujeto_id, cuerpo.grupo, cuerpo.vectores, cuerpo.modelo)
    return AltaSalida(indice=indice, sujeto_id=sujeto_id, guardados=ids)


@router.post("/indices/{indice}/sujetos/{sujeto_id}/rostros", response_model=AltaSalida)
async def alta_por_fotos(
    indice: str,
    sujeto_id: str,
    fotos: List[UploadFile] = File(...),
    grupo: str = Form(""),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    repo: Repositorio = Depends(obtener_repo),
    indices: Indices = Depends(obtener_indices),
    vision: MotorVision = Depends(obtener_vision),
):
    _validar_nombres(indice, sujeto_id)
    exigir_indice(cliente, indice)
    vectores, sin_rostro, ignoradas = [], [], 0
    for i, datos in enumerate(await _leer_fotos(fotos, config)):
        cara = vision.cara_principal(datos)
        if cara is None:
            sin_rostro.append(i)
            continue
        vectores.append(cara.vector)
        ignoradas += cara.caras_detectadas - 1
    if not vectores:
        raise ErrorV2(422, "sin_rostro", "No se encontró ninguna cara en las fotos.")
    ids = _guardar(indices, repo, indice, sujeto_id, grupo, vectores, MODELO)
    return AltaSalida(indice=indice, sujeto_id=sujeto_id, guardados=ids, sin_rostro=sin_rostro, caras_ignoradas=ignoradas)


@router.delete("/indices/{indice}/sujetos/{sujeto_id}")
def baja(
    indice: str,
    sujeto_id: str,
    cliente: Cliente = Depends(cliente_actual),
    repo: Repositorio = Depends(obtener_repo),
    indices: Indices = Depends(obtener_indices),
):
    """Borra todos los vectores del sujeto, también los que subió el teléfono."""
    _validar_nombres(indice, sujeto_id)
    exigir_indice(cliente, indice)
    borrados = repo.borrar_sujeto(indice, sujeto_id)
    indices.invalidar(indice)
    return {"indice": indice, "sujeto_id": sujeto_id, "borrados": borrados}


def _resultado(indice, res, lim, umb, tiempos, sin_rostro=None) -> ResultadoBusquedaSalida:
    return ResultadoBusquedaSalida(
        indice=indice,
        coincidencias=[
            CoincidenciaSalida(sujeto_id=c.sujeto_id, grupo=c.grupo, similitud=round(c.similitud, 4), id_vector=c.id_vector)
            for c in res.coincidencias
        ],
        comparados=res.comparados,
        umbral=umb,
        limite=lim,
        sin_rostro=sin_rostro or [],
        tiempos_ms=tiempos,
    )


@router.post("/indices/{indice}/busqueda/vector", response_model=ResultadoBusquedaSalida)
def buscar_por_vector(
    indice: str,
    cuerpo: BusquedaPorVector,
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    indices: Indices = Depends(obtener_indices),
):
    """Lo que usa la app: no sube fotos y no necesita el modelo en el servidor."""
    inicio = time.perf_counter()
    _validar_nombres(indice)
    exigir_indice(cliente, indice)
    _validar_vectores(cuerpo.modelo, cuerpo.vectores)
    lim, umb = _limites(config, cuerpo.limite, cuerpo.umbral)
    res = indices.buscar(indice, cuerpo.vectores, lim, umb, cuerpo.grupo)
    return _resultado(indice, res, lim, umb, {"total": _ms(inicio)})


@router.post("/indices/{indice}/busqueda/foto", response_model=ResultadoBusquedaSalida)
async def buscar_por_foto(
    indice: str,
    fotos: List[UploadFile] = File(...),
    limite: Optional[int] = Form(None),
    umbral: Optional[float] = Form(None),
    grupo: Optional[str] = Form(None),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    indices: Indices = Depends(obtener_indices),
    vision: MotorVision = Depends(obtener_vision),
):
    inicio = time.perf_counter()
    _validar_nombres(indice)
    exigir_indice(cliente, indice)
    lim, umb = _limites(config, limite, umbral)
    contenidos = await _leer_fotos(fotos, config)

    t = time.perf_counter()
    consultas, sin_rostro = [], []
    for i, datos in enumerate(contenidos):
        cara = vision.cara_principal(datos)
        if cara is None:
            sin_rostro.append(i)
        else:
            consultas.append(cara.vector)
    t_vision = _ms(t)
    if not consultas:
        raise ErrorV2(422, "sin_rostro", "No se encontró ninguna cara en las fotos.")

    t = time.perf_counter()
    res = indices.buscar(indice, consultas, lim, umb, grupo)
    return _resultado(
        indice, res, lim, umb, {"vision": t_vision, "busqueda": _ms(t), "total": _ms(inicio)}, sin_rostro
    )
