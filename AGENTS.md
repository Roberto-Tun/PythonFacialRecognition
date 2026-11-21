# AGENTS.md — API de reconocimiento facial (KAXTEN y KAXTEN Base)

Lee esto completo antes de tocar código. Después, en este orden:

1. `docs/v2/ESTADO.md` — qué hay hoy, qué funciona en producción y qué falta.
2. `docs/v2/API_V2.md` — referencia de los endpoints, el esquema de datos y las variables.
3. `docs/v2/PLAN_V2.md` — fases (A–E) con su estado y decisiones tomadas.
4. `docs/v2/DESPLIEGUE.md` — Cloud Run y GitHub Actions (solo para entender; no despliegues).
5. `docs/v2/PRUEBAS_LOCAL.md` — cómo correr y probar en local.

## Qué es

API en FastAPI que convierte una cara en un vector de 512 números (FaceNet) y busca
caras parecidas. Dos versiones conviven en el mismo código:

- **v1** (`/register_missing_person/`, `/identify/`): la usa la app publicada de
  **KAXTEN** (búsqueda de personas desaparecidas, en producción). No se toca.
- **v2** (`/v2/...`, rama `v2`): servicio genérico por **índices**. Un índice es un
  conjunto de caras con nombre: `miembros` (socios del check-in), `registros`
  (Registros de la plantilla del hackathon) y `desaparecidos` (los datos de KAXTEN,
  leídos en su formato de la v1). La usa **KAXTEN Base**
  (`C:\Users\Saul Colli\Documents\kaxten_base`, plantilla en Flutter) y, en el
  futuro, KAXTEN.

La v2 está **desplegada** en Cloud Run (servicio `kaxten-api-v2`, región
`northamerica-south1`, proyecto `kaxten-2090b`) y se redespliega sola con cada push
a `v2` que toque el código.

## ⛔ Reglas que no se rompen

1. **La v1 no cambia.** `/register_missing_person/` e `/identify/` siguen con las
   mismas rutas, entradas y respuestas: la app publicada de KAXTEN depende de ellas.
2. **La base de KAXTEN (`(default)`) solo se abre con la llave de KAXTEN.** Toda
   llave trabaja en la base `base`, salvo una que declare `"base_datos": "(default)"`,
   y esa solo puede tener el índice `desaparecidos` (`app/v2/config.py` se niega a
   arrancar si no). Ni el modo abierto de pruebas (`V2_SIN_LLAVE`) entra ahí. En ese
   índice la v2 **no crea personas**, filtra la visibilidad, escribe en el formato
   de la v1 (`app/v2/kaxten.py`) y exige `?confirmar=true` para borrar. Fuera de
   KAXTEN, la v2 **no sube fotos** a Storage.
3. **Mismo esquema que los teléfonos.** La v2 lee y escribe la colección `Vectores`
   de la base `base` con los campos que ya escribe la app (`indice`, `sujeto_id`,
   `grupo`, `vector`, `modelo`, `fecha`). Cambiarlo rompe el reconocimiento sin red
   de la app. Ver `docs/v2/API_V2.md`.
4. **Mismo modelo que los teléfonos.** FaceNet 20180402 (512), recorte de 160×160 y
   la normalización de `keras_facenet` (`(x − 127.5) / 127.5`, verificado en la
   imagen). Si cambia cualquiera de las tres, los vectores del servidor y los del
   teléfono dejan de ser comparables.
5. **No desplegar.** Ni `gcloud run deploy`, ni `gcloud builds submit`, ni lanzar a
   mano el flujo `.github/workflows/desplegar-v2.yml`, ni tocar la variable
   `DESPLEGAR_V2` del repositorio. **Está en `true`: cada push a `v2` que toque
   `app/`, `tests/`, `Dockerfile`, `requirements*.txt` o el flujo despliega a
   producción.** Un push solo de `docs/` no despliega. Ver `docs/v2/DESPLIEGUE.md`.
6. **No copiar llaves.** Ningún `.json` de cuenta de servicio, `.pem` ni llave de
   cliente en el código, la imagen, la documentación o un mensaje. Solo nombres de
   variables de entorno. Las llaves de clientes viven en el secreto `v2-llaves`
   (Secret Manager) y, para pruebas locales, en `.env.local` (ignorado por git y
   Docker).
7. **`config.py` tiene cambios locales de Saul.** No se commitea nunca. Si hace falta
   un ajuste, va por variable de entorno.
8. **No hacer commits ni push** salvo que Saul lo pida.
9. **No probar escrituras contra `desaparecidos`.** Es producción de KAXTEN: una
   alta la ve la app publicada. Las pruebas reales se hacen con búsquedas (solo
   leen) o en el índice `pruebas` de la base `base`.

## Forma de trabajar

- Fase por fase (`docs/v2/PLAN_V2.md`). Al cerrar cada fase: `python -m pytest -q` en
  verde y un resumen para Saul de qué cambió y qué probar.
- Español en comentarios, nombres de dominio y mensajes de error de la v2. Los
  comentarios explican **por qué**.
- Lo que no se pudo verificar (un despliegue, una base real) se dice
  explícitamente.
- En Windows, `docker run` con rutas de contenedor (`/secretos/...`) se corre desde
  **PowerShell**: Git Bash las reescribe.

## Estructura

```
app/main.py              monta la v1 (salvo V2_SOLO=1) y la v2; arregla el esquema de /docs
app/api/routes.py        v1 (KAXTEN) — no tocar salvo correcciones acordadas
app/core/                v1: detector, embedder, Firebase; credenciales.py (llave local o ADC)
app/v2/config.py         variables de entorno, llaves por cliente y guardas de KAXTEN
app/v2/auth.py           X-Api-Key → cliente → índices permitidos
app/v2/rutas.py          endpoints, contexto por cliente (base / KAXTEN), búsqueda en varios índices
app/v2/indice.py         matriz por índice en memoria, coseno, una fila por sujeto
app/v2/repositorio.py    base/Vectores (esquema del teléfono) y versión en memoria
app/v2/kaxten.py         adaptador del índice `desaparecidos` (formato v1, visibilidad, fotos)
app/v2/datos.py          datos del sujeto y foto_url en las respuestas
app/v2/vision.py         foto → cara más grande → vector (MTCNN + FaceNet, carga perezosa)
tests/                   42 pruebas sin TensorFlow ni Firebase
.github/workflows/       pruebas + imagen + Cloud Run
docs/v2/                 estado, plan, referencia, despliegue, pruebas locales
```
