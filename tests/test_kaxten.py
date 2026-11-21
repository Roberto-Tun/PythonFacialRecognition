"""Fase E: índice `desaparecidos` sobre los datos de KAXTEN (formato de la v1).

Sin Firebase: el repositorio de KAXTEN se sustituye por uno en memoria con
personas, visibilidad y fotos falsas. Lo que se prueba aquí son las reglas que
protegen producción."""

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.v2 import rutas
from app.v2.config import ConfigV2, ConfiguracionInvalida, Cliente, cargar
from app.v2.errores import ErrorV2
from app.v2.indice import Indices
from app.v2.kaxten import es_visible, sujeto_de, vector_desde_legacy
from app.v2.repositorio import RepositorioMemoria
from app.v2.vision import Cara

LLAVE_KAXTEN = "llave-de-prueba-kaxten-0123456789"
LLAVE_DEMO = "llave-de-prueba-demo-0123456789"


def vec(semilla: int) -> list:
    return [float(x) for x in np.random.default_rng(semilla).normal(size=512)]


class KaxtenFalso(RepositorioMemoria):
    """Repositorio en memoria + lo propio de KAXTEN."""

    def __init__(self, personas: dict):
        super().__init__()
        self.personas = personas  # id → datos (con o sin "visibilidad")
        self.fotos: list = []

    def existe(self, sujeto_id):
        return sujeto_id in self.personas

    def visibles(self, sujetos):
        return {s for s in sujetos if es_visible(self.personas.get(s))}

    def guardar(self, vectores):
        # Como RepositorioKaxten: el formato de la v1 sí lleva `ruta_storage`.
        for v in vectores:
            self.docs[v.id] = {**v.documento(), "ruta_storage": v.ruta_storage}

    def leer_indice(self, indice):
        from app.v2.repositorio import Vector

        self.lecturas += 1
        return [
            Vector(**{**vars(Vector.desde_documento(i, d)), "ruta_storage": d.get("ruta_storage", "")})
            for i, d in self.docs.items()
            if d.get("indice") == indice
        ]

    def datos(self, indice, ids):
        return {i: self.personas[i] for i in ids if i in self.personas}

    def subir_foto(self, sujeto_id, contenido):
        self.fotos.append(sujeto_id)
        return f"https://storage.googleapis.com/bucket/rostros/{sujeto_id}.jpg"


class VisionFalsa:
    def cara_principal(self, contenido: bytes):
        texto = contenido.decode(errors="ignore")
        if not texto.startswith("cara:"):
            return None
        return Cara(vector=vec(int(texto[5:])), caja=(0, 0, 10, 10), confianza=0.99, caras_detectadas=1)


@pytest.fixture
def entorno():
    kaxten = KaxtenFalso(
        {
            "publica": {"nombre": "A"},  # sin campo: cuenta como pública
            "publica2": {"visibilidad": 1},
            "desactivada": {"visibilidad": 2},
            "retirada": {"visibilidad": 3},
            "rara": {"visibilidad": "x"},
        }
    )
    base = RepositorioMemoria()
    cfg = ConfigV2(
        llaves={
            LLAVE_KAXTEN: Cliente("kaxten", frozenset({"desaparecidos"}), base_datos="(default)"),
            LLAVE_DEMO: Cliente("kaxten_base", frozenset({"miembros"})),
        },
        refresco_s=3600,
    )
    ctx_k = rutas.Contexto(repo=kaxten, indices=Indices(kaxten, 3600), kaxten=kaxten)
    ctx_b = rutas.Contexto(repo=base, indices=Indices(base, 3600))

    app = FastAPI()
    app.include_router(rutas.router)

    @app.exception_handler(ErrorV2)
    async def _h(_: Request, e: ErrorV2):
        return JSONResponse(status_code=e.estado, content=e.cuerpo())

    def contexto(cliente: Cliente = rutas.Depends(rutas.cliente_actual)):
        return ctx_k if cliente.es_kaxten else ctx_b

    app.dependency_overrides[rutas.obtener_config] = lambda: cfg
    app.dependency_overrides[rutas.obtener_contexto] = contexto
    app.dependency_overrides[rutas.obtener_vision] = lambda: VisionFalsa()
    return TestClient(app), kaxten, base


HK = {"X-Api-Key": LLAVE_KAXTEN}
HD = {"X-Api-Key": LLAVE_DEMO}


def alta(cli, persona, v, h=HK):
    return cli.post(
        f"/v2/indices/desaparecidos/sujetos/{persona}/vectores",
        json={"modelo": "facenet_512", "vectores": [v]},
        headers=h,
    )


def buscar(cli, v, h=HK, **extra):
    return cli.post(
        "/v2/indices/desaparecidos/busqueda/vector", json={"modelo": "facenet_512", "vectores": [v], **extra}, headers=h
    )


# --- Configuración: la base de KAXTEN solo con su llave ----------------------------


def test_llave_de_kaxten_solo_con_su_indice():
    with pytest.raises(ConfiguracionInvalida):
        cargar({"V2_LLAVES": f'{{"{LLAVE_KAXTEN}": {{"cliente": "k", "base_datos": "(default)", "indices": ["desaparecidos", "miembros"]}}}}'})
    with pytest.raises(ConfiguracionInvalida):
        cargar({"V2_LLAVES": f'{{"{LLAVE_DEMO}": {{"cliente": "d", "indices": ["desaparecidos"]}}}}'})
    with pytest.raises(ConfiguracionInvalida):
        cargar({"V2_LLAVES": f'{{"{LLAVE_DEMO}": {{"cliente": "d", "indices": ["*"]}}}}'})
    cfg = cargar({"V2_LLAVES": f'{{"{LLAVE_KAXTEN}": {{"cliente": "k", "base_datos": "(default)", "indices": ["desaparecidos"]}}}}'})
    assert next(iter(cfg.llaves.values())).es_kaxten
    # La base global de la v2 sigue sin poder ser la de KAXTEN.
    assert cfg.base_datos == "base"


def test_otras_llaves_no_entran_a_kaxten(entorno):
    cli, _, _ = entorno
    assert buscar(cli, vec(1), h=HD).status_code == 403
    assert alta(cli, "publica", vec(1), h=HD).status_code == 403


def test_modo_abierto_local_no_entra_a_kaxten():
    from app.v2.auth import CLIENTE_LOCAL, exigir_indice

    with pytest.raises(ErrorV2) as e:
        exigir_indice(CLIENTE_LOCAL, "desaparecidos")
    assert e.value.estado == 403


def test_llave_de_kaxten_no_entra_a_otros_indices(entorno):
    cli, _, _ = entorno
    r = cli.post("/v2/indices/miembros/busqueda/vector", json={"modelo": "facenet_512", "vectores": [vec(1)]}, headers=HK)
    assert r.status_code == 403


# --- Reglas del índice ---------------------------------------------------------------


def test_solo_coinciden_reportes_publicos(entorno):
    cli, _, _ = entorno
    base = vec(1)
    for p in ("publica", "publica2", "desactivada", "retirada", "rara"):
        assert alta(cli, p, base).status_code == 200
    r = buscar(cli, base)
    assert r.status_code == 200
    assert sorted(c["sujeto_id"] for c in r.json()["coincidencias"]) == ["publica", "publica2"]


def test_no_se_dan_de_alta_personas_que_no_existen(entorno):
    cli, kaxten, _ = entorno
    r = alta(cli, "inventada", vec(1))
    assert r.status_code == 404
    assert r.json()["error"] == "sujeto_desconocido"
    assert kaxten.docs == {}


def test_baja_exige_confirmar(entorno):
    cli, kaxten, _ = entorno
    alta(cli, "publica", vec(1))
    r = cli.delete("/v2/indices/desaparecidos/sujetos/publica", headers=HK)
    assert r.status_code == 400
    assert r.json()["error"] == "confirmacion_requerida"
    assert len(kaxten.docs) == 1
    r = cli.delete("/v2/indices/desaparecidos/sujetos/publica?confirmar=true", headers=HK)
    assert r.json()["borrados"] == 1


def test_alta_por_foto_guarda_la_foto_como_la_v1(entorno):
    cli, kaxten, _ = entorno
    r = cli.post(
        "/v2/indices/desaparecidos/sujetos/publica/rostros",
        files=[("fotos", ("a.jpg", b"cara:4", "image/jpeg"))],
        headers=HK,
    )
    assert r.status_code == 200
    assert kaxten.fotos == ["publica"]


def test_la_demo_no_sube_fotos(entorno):
    cli, kaxten, base = entorno
    r = cli.post(
        "/v2/indices/miembros/sujetos/s1/rostros", files=[("fotos", ("a.jpg", b"cara:4", "image/jpeg"))], headers=HD
    )
    assert r.status_code == 200
    assert kaxten.fotos == [] and len(base.docs) == 1


# --- Formato de la v1 ---------------------------------------------------------------


class _Ref:
    def __init__(self, id_):
        self.id = id_


def test_lee_referencias_y_ids_viejos():
    assert sujeto_de(_Ref("abc")) == "abc"
    assert sujeto_de("PersonasDesaparecidas/abc") == "abc"
    assert sujeto_de("abc") == "abc"
    assert sujeto_de(None) == ""


def test_vector_de_la_v1_sin_modelo_es_del_servidor():
    v = vector_desde_legacy("x", {"id_persona_desaparecida": _Ref("p1"), "vector": vec(1), "ruta_storage": "u"})
    assert v.indice == "desaparecidos" and v.sujeto_id == "p1" and v.modelo == "facenet_512" and v.ruta_storage == "u"


def test_visibilidad_como_la_app():
    assert es_visible({}) and es_visible({"visibilidad": 1}) and es_visible({"visibilidad": "1"})
    assert not es_visible({"visibilidad": 2}) and not es_visible({"visibilidad": 3})
    assert not es_visible({"visibilidad": "x"}) and not es_visible(None)


def test_kaxten_devuelve_la_persona_y_su_foto(entorno):
    cli, kaxten, _ = entorno
    kaxten.personas["publica"]["nombre"] = "Juan"
    r = cli.post(
        "/v2/indices/desaparecidos/sujetos/publica/rostros", files=[("fotos", ("a.jpg", b"cara:4", "image/jpeg"))], headers=HK
    )
    assert r.status_code == 200
    c = cli.post("/v2/busqueda/foto", files=[("fotos", ("q.jpg", b"cara:4", "image/jpeg"))], headers=HK).json()["coincidencias"][0]
    assert c["indice"] == "desaparecidos" and c["datos"]["nombre"] == "Juan"
    assert c["foto_url"].endswith("/rostros/publica.jpg")
