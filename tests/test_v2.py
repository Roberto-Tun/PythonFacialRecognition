"""Pruebas de la v2 sin TensorFlow ni Firebase: repositorio en memoria y un motor
de visión falso. Correr con `python -m pytest tests -q`."""

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.v2 import rutas
from app.v2.config import ConfigV2, ConfiguracionInvalida, Cliente, cargar
from app.v2.errores import ErrorV2
from app.v2.indice import Indices
from app.v2.repositorio import RepositorioMemoria
from app.v2.vision import Cara

LLAVE_DEMO = "llave-de-prueba-demo-0123456789"
LLAVE_OTRO = "llave-de-prueba-otro-0123456789"


def vec(semilla: int) -> list:
    rng = np.random.default_rng(semilla)
    return [float(x) for x in rng.normal(size=512)]


def cerca(v: list, ruido: float = 0.05, semilla: int = 99) -> list:
    rng = np.random.default_rng(semilla)
    return [float(a + b) for a, b in zip(v, rng.normal(scale=ruido, size=512))]


class VisionFalsa:
    """b"cara:<n>" → vector de la semilla n; cualquier otra cosa → sin cara."""

    def cara_principal(self, contenido: bytes):
        texto = contenido.decode(errors="ignore")
        if not texto.startswith("cara:"):
            return None
        return Cara(vector=vec(int(texto[5:])), caja=(0, 0, 10, 10), confianza=0.99, caras_detectadas=1)


@pytest.fixture
def entorno():
    repo = RepositorioMemoria()
    cfg = ConfigV2(
        llaves={
            LLAVE_DEMO: Cliente("kaxten_base", frozenset({"miembros", "registros"})),
            LLAVE_OTRO: Cliente("otro", frozenset({"clientes"})),
        },
        refresco_s=3600,
    )
    indices = Indices(repo, cfg.refresco_s)
    app = FastAPI()
    app.include_router(rutas.router)

    @app.exception_handler(ErrorV2)
    async def _h(_: Request, e: ErrorV2):
        return JSONResponse(status_code=e.estado, content=e.cuerpo())

    app.dependency_overrides[rutas.obtener_config] = lambda: cfg
    app.dependency_overrides[rutas.obtener_repo] = lambda: repo
    app.dependency_overrides[rutas.obtener_indices] = lambda: indices
    app.dependency_overrides[rutas.obtener_vision] = lambda: VisionFalsa()
    return TestClient(app), repo


H = {"X-Api-Key": LLAVE_DEMO}


def alta(cli, sujeto, v, grupo="", indice="miembros"):
    return cli.post(
        f"/v2/indices/{indice}/sujetos/{sujeto}/vectores",
        json={"modelo": "facenet_512_fp16", "vectores": [v], "grupo": grupo},
        headers=H,
    )


def buscar(cli, v, **extra):
    cuerpo = {"modelo": "facenet_512_fp16", "vectores": [v], **extra}
    return cli.post("/v2/indices/miembros/busqueda/vector", json=cuerpo, headers=H)


# --- Configuración y acceso -------------------------------------------------------


def test_no_arranca_contra_la_base_de_kaxten():
    for valor in ("(default)", "default", ""):
        with pytest.raises(ConfiguracionInvalida):
            cargar({"V2_BASE_DATOS": valor})
    assert cargar({}).base_datos == "base"


def test_llaves_cortas_o_mal_escritas_no_arrancan():
    with pytest.raises(ConfiguracionInvalida):
        cargar({"V2_LLAVES": '{"corta": {"cliente": "x", "indices": ["a"]}}'})
    with pytest.raises(ConfiguracionInvalida):
        cargar({"V2_LLAVES": "{no es json"})


def test_salud_sin_llave(entorno):
    cli, _ = entorno
    r = cli.get("/v2/salud")
    assert r.status_code == 200
    assert r.json()["base_datos"] == "base"
    assert r.json()["autenticacion"] == "llaves"


def test_sin_llave_401_y_indice_ajeno_403(entorno):
    cli, _ = entorno
    cuerpo = {"modelo": "facenet_512", "vectores": [vec(1)]}
    assert cli.post("/v2/indices/miembros/busqueda/vector", json=cuerpo).status_code == 401
    r = cli.post("/v2/indices/miembros/busqueda/vector", json=cuerpo, headers={"X-Api-Key": LLAVE_OTRO})
    assert r.status_code == 403
    assert r.json()["error"] == "indice_ajeno"


def test_sin_llaves_configuradas_responde_503_salvo_modo_local():
    from app.v2.auth import identificar

    with pytest.raises(ErrorV2) as e:
        identificar(ConfigV2(), None)
    assert e.value.estado == 503
    assert identificar(ConfigV2(sin_llave=True), None).puede("cualquiera")


# --- Alta y búsqueda ---------------------------------------------------------------


def test_alta_y_busqueda_por_vector(entorno):
    cli, _ = entorno
    ana, beto = vec(1), vec(2)
    assert alta(cli, "ana", ana).status_code == 200
    assert alta(cli, "beto", beto).status_code == 200

    r = buscar(cli, cerca(ana))
    assert r.status_code == 200
    c = r.json()["coincidencias"]
    assert [x["sujeto_id"] for x in c] == ["ana"]
    assert c[0]["similitud"] > 0.95
    assert r.json()["comparados"] == 2


def test_cara_desconocida_no_devuelve_al_mas_parecido(entorno):
    cli, _ = entorno
    alta(cli, "ana", vec(1))
    r = buscar(cli, vec(500))
    assert r.json()["coincidencias"] == []


def test_una_fila_por_sujeto(entorno):
    cli, _ = entorno
    ana = vec(1)
    cli.post(
        "/v2/indices/miembros/sujetos/ana/vectores",
        json={"modelo": "facenet_512", "vectores": [ana, cerca(ana, semilla=5), cerca(ana, semilla=6)]},
        headers=H,
    )
    r = buscar(cli, cerca(ana))
    assert len(r.json()["coincidencias"]) == 1


def test_grupo_separa_negocios(entorno):
    cli, _ = entorno
    ana = vec(1)
    alta(cli, "ana", ana, grupo="demo_gimnasio")
    assert buscar(cli, ana, grupo="demo_restaurante").json()["coincidencias"] == []
    assert len(buscar(cli, ana, grupo="demo_gimnasio").json()["coincidencias"]) == 1


def test_umbral_y_limite(entorno):
    cli, _ = entorno
    base = vec(1)
    for i in range(5):
        alta(cli, f"s{i}", cerca(base, ruido=0.3, semilla=i))
    r = buscar(cli, base, limite=2, umbral=0.0)
    assert len(r.json()["coincidencias"]) == 2
    assert buscar(cli, base, umbral=2).status_code == 400


def test_modelo_incompatible_y_vector_invalido(entorno):
    cli, _ = entorno
    r = cli.post(
        "/v2/indices/miembros/busqueda/vector", json={"modelo": "arcface", "vectores": [vec(1)]}, headers=H
    )
    assert r.status_code == 409
    r = cli.post(
        "/v2/indices/miembros/busqueda/vector", json={"modelo": "facenet_512", "vectores": [[0.1] * 511]}, headers=H
    )
    assert r.status_code == 400
    assert r.json()["error"] == "vector_invalido"


def test_nombres_invalidos(entorno):
    cli, _ = entorno
    assert alta(cli, "ana", vec(1), indice="Miembros!").status_code == 400
    assert alta(cli, "../ana", vec(1)).status_code in (400, 404)


# --- Compatibilidad con los teléfonos ---------------------------------------------


def test_esquema_igual_al_del_telefono(entorno):
    cli, repo = entorno
    alta(cli, "ana", vec(1), grupo="demo_gimnasio")
    (doc,) = repo.docs.values()
    assert set(doc) == {"indice", "sujeto_id", "grupo", "vector", "modelo", "fecha"}
    assert doc["indice"] == "miembros" and doc["sujeto_id"] == "ana" and len(doc["vector"]) == 512
    assert isinstance(doc["fecha"], str)


def test_encuentra_lo_que_subio_un_telefono(entorno):
    cli, repo = entorno
    # Así escribe la app (`VectorFacial.aMapa`), con su id `<indice>_<sujeto>_<n>`.
    repo.docs["miembros_socio123_0"] = {
        "indice": "miembros",
        "sujeto_id": "socio123",
        "grupo": "demo_gimnasio",
        "vector": vec(7),
        "modelo": "facenet_512_fp16",
        "fecha": "2026-11-20T10:00:00.000",
    }
    r = buscar(cli, cerca(vec(7)))
    assert r.json()["coincidencias"][0]["sujeto_id"] == "socio123"


def test_ignora_vectores_de_otro_modelo(entorno):
    cli, repo = entorno
    repo.docs["miembros_x_0"] = {"indice": "miembros", "sujeto_id": "x", "grupo": "", "vector": vec(7), "modelo": "otro"}
    assert buscar(cli, vec(7)).json()["comparados"] == 0


def test_alta_idempotente(entorno):
    cli, repo = entorno
    a = alta(cli, "ana", vec(1)).json()["guardados"]
    b = alta(cli, "ana", vec(1)).json()["guardados"]
    assert a == b and len(repo.docs) == 1


def test_baja_borra_tambien_lo_del_telefono(entorno):
    cli, repo = entorno
    alta(cli, "ana", vec(1))
    repo.docs["miembros_ana_0"] = {**next(iter(repo.docs.values()))}
    r = cli.delete("/v2/indices/miembros/sujetos/ana", headers=H)
    assert r.json()["borrados"] == 2
    assert buscar(cli, vec(1)).json()["coincidencias"] == []


def test_la_cache_no_relee_toda_la_coleccion(entorno):
    cli, repo = entorno
    alta(cli, "ana", vec(1))
    buscar(cli, vec(1))
    lecturas = repo.lecturas
    for _ in range(5):
        buscar(cli, vec(1))
    assert repo.lecturas == lecturas


# --- Por foto ---------------------------------------------------------------------


def test_alta_y_busqueda_por_foto(entorno):
    cli, _ = entorno
    r = cli.post(
        "/v2/indices/registros/sujetos/r1/rostros",
        files=[("fotos", ("a.jpg", b"cara:3", "image/jpeg")), ("fotos", ("b.jpg", b"paisaje", "image/jpeg"))],
        headers=H,
    )
    assert r.status_code == 200
    assert r.json()["sin_rostro"] == [1]
    r = cli.post(
        "/v2/indices/registros/busqueda/foto", files=[("fotos", ("q.jpg", b"cara:3", "image/jpeg"))], headers=H
    )
    assert r.json()["coincidencias"][0]["sujeto_id"] == "r1"
    assert "vision" in r.json()["tiempos_ms"]


def test_foto_sin_cara(entorno):
    cli, _ = entorno
    r = cli.post(
        "/v2/indices/registros/sujetos/r1/rostros", files=[("fotos", ("a.jpg", b"paisaje", "image/jpeg"))], headers=H
    )
    assert r.status_code == 422
    assert r.json()["error"] == "sin_rostro"
