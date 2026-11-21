"""Endpoints de la v2. Referencia para quien la consume: docs/v2/API_V2.md."""

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, Security, UploadFile
from fastapi.security import APIKeyHeader

from app.v2 import VERSION
from app.v2.auth import exigir_indice, identificar
from app.v2.config import BASE_KAXTEN, DIMENSION, INDICE_KAXTEN, MODELO, MODELOS_COMPATIBLES, Cliente, ConfigV2, cargar
from app.v2.datos import foto_de
from app.v2.errores import ErrorV2
from app.v2.esquemas import (
    LIMITE_DEFECTO,
    UMBRAL_DEFECTO,
    AltaPorVectores,
    AltaSalida,
    BusquedaPorVector,
    CoincidenciaSalida,
    ResultadoBusquedaSalida,
)
from app.v2.indice import Indices, ResultadoBusqueda, vector_valido
from app.v2.repositorio import Repositorio, Vector, ahora_iso, id_servidor
from app.v2.vision import MotorVision

router = APIRouter(prefix="/v2", tags=["v2"])

_NOMBRE_INDICE = re.compile(r"^[a-z0-9_]{1,40}$")
# Ids de Firestore generados por la app (20 caracteres) o del dominio del cliente.
_ID_SUJETO = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


@dataclass
class Contexto:
    """Dónde trabaja una petición. `kaxten` solo existe para la llave de KAXTEN:
    agrega la visibilidad, la comprobación de que la persona existe y la foto en
    Storage (ver app/v2/kaxten.py)."""

    repo: Repositorio
    indices: Indices
    kaxten: Optional[object] = None


# --- Dependencias (las pruebas las reemplazan con dependency_overrides) --------


@lru_cache
def obtener_config() -> ConfigV2:
    return cargar()


def _sin_credenciales(e: Exception) -> ErrorV2:
    # El tipo sí, el texto no: puede traer rutas de la máquina.
    return ErrorV2(
        503,
        "sin_credenciales",
        f"El servidor no tiene credenciales de Google ({type(e).__name__}). "
        "En local: monta la llave y define GOOGLE_APPLICATION_CREDENTIALS (docs/v2/PRUEBAS_LOCAL.md).",
    )


@lru_cache
def _contexto_base() -> Contexto:
    from firebase_admin import firestore

    from app.core.credenciales import inicializar_firebase
    from app.v2.repositorio import RepositorioFirestore

    cfg = obtener_config()
    try:
        cliente = firestore.client(app=inicializar_firebase(), database_id=cfg.base_datos)
    except Exception as e:  # DefaultCredentialsError, archivo ilegible…
        raise _sin_credenciales(e) from None
    repo = RepositorioFirestore(cliente, cfg.coleccion)
    return Contexto(repo=repo, indices=Indices(repo, cfg.refresco_s))


@lru_cache
def _contexto_kaxten() -> Contexto:
    from firebase_admin import firestore, storage

    from app.core.credenciales import inicializar_firebase
    from app.v2.kaxten import RepositorioKaxten

    try:
        app = inicializar_firebase()
        repo = RepositorioKaxten(firestore.client(app=app), storage.bucket(app=app))
    except Exception as e:
        raise _sin_credenciales(e) from None
    # KAXTEN tiene miles de rostros: recargarlos cada 30 s costaría segundos y
    # miles de lecturas. Sus altas por la v2 invalidan la caché al momento; lo
    # que suba la v1 se ve al siguiente refresco.
    return Contexto(repo=repo, indices=Indices(repo, obtener_config().refresco_kaxten_s), kaxten=repo)


@lru_cache
def obtener_vision() -> MotorVision:
    from app.v2.vision import MotorFaceNet

    return MotorFaceNet()


# Como esquema de seguridad (y no como un encabezado suelto) para que `/docs`
# muestre el botón "Authorize": la llave se escribe una vez y va en todas las
# peticiones.
_esquema_llave = APIKeyHeader(name="X-Api-Key", auto_error=False, description="Llave del cliente (V2_LLAVES)")


def cliente_actual(
    x_api_key: Optional[str] = Security(_esquema_llave),
    config: ConfigV2 = Depends(obtener_config),
) -> Cliente:
    return identificar(config, x_api_key)


def obtener_contexto(cliente: Cliente = Depends(cliente_actual)) -> Contexto:
    return _contexto_kaxten() if cliente.base_datos == BASE_KAXTEN else _contexto_base()


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
    lim = LIMITE_DEFECTO if limite is None else max(1, min(int(limite), config.limite_max))
    umb = UMBRAL_DEFECTO if umbral is None else float(umbral)
    if not 0.0 <= umb <= 1.0:
        raise ErrorV2(400, "umbral_invalido", "El umbral va de 0 a 1.")
    return lim, umb


def _exigir_persona(ctx: Contexto, sujeto_id: str) -> None:
    """En KAXTEN la v2 no crea personas: un vector sin persona sería un huérfano
    en producción."""
    if ctx.kaxten is not None and not ctx.kaxten.existe(sujeto_id):
        raise ErrorV2(404, "sujeto_desconocido", f"No existe la persona '{sujeto_id}' en PersonasDesaparecidas.")


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


def _buscar(ctx: Contexto, indice: str, consultas, lim: int, umb: float, grupo: Optional[str]) -> ResultadoBusqueda:
    if ctx.kaxten is None:
        return ctx.indices.buscar(indice, consultas, lim, umb, grupo)
    # KAXTEN (T-05): una persona con el reporte oculto no es coincidencia. Se
    # piden más candidatos de los que se devuelven para no quedarse cortos
    # después de filtrar.
    res = ctx.indices.buscar(indice, consultas, min(lim * 3, 200), umb, None)
    visibles = ctx.kaxten.visibles(c.sujeto_id for c in res.coincidencias)
    return ResultadoBusqueda([c for c in res.coincidencias if c.sujeto_id in visibles][:lim], res.comparados)


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
        "kaxten": any(c.base_datos == BASE_KAXTEN for c in config.llaves.values()),
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


def _guardar(ctx: Contexto, indice: str, sujeto_id: str, grupo: str, vectores, modelo: str, fotos=None):
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
            ruta_storage=(fotos[i] if fotos else ""),
        )
        for i, v in enumerate(vectores)
    ]
    ctx.repo.guardar(docs)
    ctx.indices.invalidar(indice)
    # Sin repetir: la misma cara enviada dos veces es un solo documento.
    return list(dict.fromkeys(d.id for d in docs))


@router.post("/indices/{indice}/sujetos/{sujeto_id}/vectores", response_model=AltaSalida)
def alta_por_vectores(
    indice: str,
    sujeto_id: str,
    cuerpo: AltaPorVectores,
    cliente: Cliente = Depends(cliente_actual),
    ctx: Contexto = Depends(obtener_contexto),
):
    """Lo que usa la app: el teléfono ya calculó los vectores."""
    _validar_nombres(indice, sujeto_id)
    exigir_indice(cliente, indice)
    _validar_vectores(cuerpo.modelo, cuerpo.vectores)
    _exigir_persona(ctx, sujeto_id)
    ids = _guardar(ctx, indice, sujeto_id, cuerpo.grupo, cuerpo.vectores, cuerpo.modelo)
    return AltaSalida(indice=indice, sujeto_id=sujeto_id, guardados=ids)


@router.post("/indices/{indice}/sujetos/{sujeto_id}/rostros", response_model=AltaSalida)
async def alta_por_fotos(
    indice: str,
    sujeto_id: str,
    fotos: List[UploadFile] = File(...),
    grupo: str = Form(""),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    ctx: Contexto = Depends(obtener_contexto),
    vision: MotorVision = Depends(obtener_vision),
):
    _validar_nombres(indice, sujeto_id)
    exigir_indice(cliente, indice)
    _exigir_persona(ctx, sujeto_id)
    vectores, con_cara, sin_rostro, ignoradas = [], [], [], 0
    for i, datos in enumerate(await _leer_fotos(fotos, config)):
        cara = vision.cara_principal(datos)
        if cara is None:
            sin_rostro.append(i)
            continue
        vectores.append(cara.vector)
        con_cara.append(datos)
        ignoradas += cara.caras_detectadas - 1
    if not vectores:
        raise ErrorV2(422, "sin_rostro", "No se encontró ninguna cara en las fotos.")
    # Solo KAXTEN guarda la foto: su app la muestra junto a la persona. La demo
    # y los demás clientes guardan solo vectores.
    rutas = [ctx.kaxten.subir_foto(sujeto_id, d) for d in con_cara] if ctx.kaxten is not None else None
    ids = _guardar(ctx, indice, sujeto_id, grupo, vectores, MODELO, rutas)
    return AltaSalida(indice=indice, sujeto_id=sujeto_id, guardados=ids, sin_rostro=sin_rostro, caras_ignoradas=ignoradas)


@router.delete("/indices/{indice}/sujetos/{sujeto_id}")
def baja(
    indice: str,
    sujeto_id: str,
    confirmar: bool = Query(False, description="Obligatorio para el índice de KAXTEN"),
    cliente: Cliente = Depends(cliente_actual),
    ctx: Contexto = Depends(obtener_contexto),
):
    """Borra todos los vectores del sujeto, también los que subió el teléfono."""
    _validar_nombres(indice, sujeto_id)
    exigir_indice(cliente, indice)
    if ctx.kaxten is not None and not confirmar:
        raise ErrorV2(
            400,
            "confirmacion_requerida",
            "Esto borra los vectores de una persona en producción. Repite con ?confirmar=true.",
        )
    borrados = ctx.repo.borrar_sujeto(indice, sujeto_id)
    ctx.indices.invalidar(indice)
    return {"indice": indice, "sujeto_id": sujeto_id, "borrados": borrados}


def _datos_de(ctx: Contexto, coincidencias, con_datos: bool) -> dict:
    """(índice, sujeto) → datos. Una lectura por índice, solo de los resultados
    finales: nunca de candidatos descartados."""
    if not con_datos:
        return {}
    por_indice: dict = {}
    for i, c in coincidencias:
        por_indice.setdefault(i, []).append(c.sujeto_id)
    salida = {}
    for i, ids in por_indice.items():
        try:
            for sujeto, d in ctx.repo.datos(i, list(dict.fromkeys(ids))).items():
                salida[(i, sujeto)] = d
        except Exception as e:  # sin los datos la búsqueda sigue sirviendo
            print(f"v2: no se pudieron leer los datos de '{i}': {type(e).__name__}")
    return salida


def _resultado(indices: List[str], coincidencias, comparados: int, lim: int, umb: float, tiempos, sin_rostro=None, datos=None):
    datos = datos or {}
    return ResultadoBusquedaSalida(
        indice=",".join(indices),
        coincidencias=[
            CoincidenciaSalida(
                indice=i,
                sujeto_id=c.sujeto_id,
                grupo=c.grupo,
                similitud=round(c.similitud, 4),
                id_vector=c.id_vector,
                foto_url=foto_de(i, datos.get((i, c.sujeto_id)), c.ruta_storage),
                datos=datos.get((i, c.sujeto_id)),
            )
            for i, c in coincidencias
        ],
        comparados=comparados,
        umbral=umb,
        limite=lim,
        sin_rostro=sin_rostro or [],
        tiempos_ms=tiempos,
    )


def _indices_a_buscar(ctx: Contexto, cliente: Cliente, indice: Optional[str]) -> List[str]:
    """El índice pedido o, si no se dice, todos los que la llave puede ver."""
    if indice:
        _validar_nombres(indice)
        exigir_indice(cliente, indice)
        return [indice]
    if cliente.es_kaxten:
        return [INDICE_KAXTEN]
    candidatos = ctx.repo.indices_existentes() if "*" in cliente.indices else sorted(cliente.indices)
    permitidos = [
        i for i in candidatos if i != INDICE_KAXTEN and _NOMBRE_INDICE.match(i) and cliente.puede(i)
    ]
    if not permitidos:
        raise ErrorV2(404, "sin_indices", "Todavía no hay caras guardadas en ningún índice de este cliente.")
    return permitidos


def _buscar_en_varios(ctx: Contexto, indices: List[str], consultas, lim: int, umb: float, grupo: Optional[str]):
    juntas, comparados = [], 0
    for i in indices:
        res = _buscar(ctx, i, consultas, lim, umb, grupo)
        comparados += res.comparados
        juntas.extend((i, c) for c in res.coincidencias)
    juntas.sort(key=lambda par: par[1].similitud, reverse=True)
    return juntas[:lim], comparados


def _por_vector(indice, cuerpo: BusquedaPorVector, cliente, config, ctx, con_datos=True) -> ResultadoBusquedaSalida:
    inicio = time.perf_counter()
    indices = _indices_a_buscar(ctx, cliente, indice)
    _validar_vectores(cuerpo.modelo, cuerpo.vectores)
    lim, umb = _limites(config, cuerpo.limite, cuerpo.umbral)
    coincidencias, comparados = _buscar_en_varios(ctx, indices, cuerpo.vectores, lim, umb, cuerpo.grupo or None)
    datos = _datos_de(ctx, coincidencias, con_datos)
    return _resultado(indices, coincidencias, comparados, lim, umb, {"total": _ms(inicio)}, datos=datos)


async def _por_foto(indice, fotos, limite, umbral, grupo, cliente, config, ctx, vision, con_datos=True) -> ResultadoBusquedaSalida:
    inicio = time.perf_counter()
    indices = _indices_a_buscar(ctx, cliente, indice)
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
    coincidencias, comparados = _buscar_en_varios(ctx, indices, consultas, lim, umb, grupo or None)
    t_busqueda = _ms(t)
    t = time.perf_counter()
    datos = _datos_de(ctx, coincidencias, con_datos)
    return _resultado(
        indices, coincidencias, comparados, lim, umb,
        {"vision": t_vision, "busqueda": t_busqueda, "datos": _ms(t), "total": _ms(inicio)}, sin_rostro, datos,
    )


@router.post("/busqueda/foto", response_model=ResultadoBusquedaSalida)
async def buscar_por_foto_sin_indice(
    fotos: List[UploadFile] = File(..., description="1 a 5 fotos"),
    indice: str = Form("", description="Opcional. Vacío: busca en todos los índices de la llave"),
    limite: int = Form(LIMITE_DEFECTO, ge=1),
    umbral: float = Form(UMBRAL_DEFECTO, ge=0, le=1),
    grupo: str = Form("", description="Opcional. Vacío: todos los grupos"),
    con_datos: bool = Form(True, description="Incluir datos y foto de cada sujeto"),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    ctx: Contexto = Depends(obtener_contexto),
    vision: MotorVision = Depends(obtener_vision),
):
    """Sube una foto y busca esa cara. Lo más simple para probar."""
    return await _por_foto(indice, fotos, limite, umbral, grupo, cliente, config, ctx, vision, con_datos)


@router.post("/busqueda/vector", response_model=ResultadoBusquedaSalida)
def buscar_por_vector_sin_indice(
    cuerpo: BusquedaPorVector,
    indice: str = Query("", description="Opcional. Vacío: busca en todos los índices de la llave"),
    con_datos: bool = Query(True, description="Incluir datos y foto de cada sujeto"),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    ctx: Contexto = Depends(obtener_contexto),
):
    return _por_vector(indice, cuerpo, cliente, config, ctx, con_datos)


@router.post("/indices/{indice}/busqueda/vector", response_model=ResultadoBusquedaSalida)
def buscar_por_vector(
    indice: str,
    cuerpo: BusquedaPorVector,
    con_datos: bool = Query(True, description="Incluir datos y foto de cada sujeto"),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    ctx: Contexto = Depends(obtener_contexto),
):
    """Lo que usa la app: no sube fotos y no necesita el modelo en el servidor."""
    return _por_vector(indice, cuerpo, cliente, config, ctx, con_datos)


@router.post("/indices/{indice}/busqueda/foto", response_model=ResultadoBusquedaSalida)
async def buscar_por_foto(
    indice: str,
    fotos: List[UploadFile] = File(..., description="1 a 5 fotos"),
    limite: int = Form(LIMITE_DEFECTO, ge=1),
    umbral: float = Form(UMBRAL_DEFECTO, ge=0, le=1),
    grupo: str = Form("", description="Opcional. Vacío: todos los grupos"),
    con_datos: bool = Form(True, description="Incluir datos y foto de cada sujeto"),
    cliente: Cliente = Depends(cliente_actual),
    config: ConfigV2 = Depends(obtener_config),
    ctx: Contexto = Depends(obtener_contexto),
    vision: MotorVision = Depends(obtener_vision),
):
    return await _por_foto(indice, fotos, limite, umbral, grupo, cliente, config, ctx, vision, con_datos)
