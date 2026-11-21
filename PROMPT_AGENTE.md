# Prompt para un agente nuevo

Copia todo lo que está entre las líneas `---` y pégalo como primer mensaje.

---

Hola. Vas a trabajar en la **API de reconocimiento facial** de mi equipo (Project
ResQ, Tecnológico de Chetumal). Soy Saul. Trabajamos en español: código de
dominio, comentarios y mensajes en español.

**El proyecto**

- Repo: `C:\Users\Saul Colli\Documents\PythonFacialRecognition`, rama **`v2`**.
  FastAPI + TensorFlow (MTCNN + FaceNet 512) + Firestore.
- La **v1** (`/identify/`, `/register_missing_person/`) la usa **KAXTEN**, nuestra
  app publicada de búsqueda de personas desaparecidas. **No se toca.**
- La **v2** (`/v2/...`) es un servicio genérico por **índices** (conjuntos de
  caras): `miembros` y `registros` para **KAXTEN Base**, nuestra plantilla Flutter
  para el hackathon (`C:\Users\Saul Colli\Documents\kaxten_base`), y
  `desaparecidos` para KAXTEN (sus datos de producción, leídos sin migrar).
- La v2 está **en producción** en Cloud Run (`kaxten-api-v2`) y **se despliega sola
  con cada push a `v2` que toque el código** (GitHub Actions).
- Los teléfonos reconocen sin red con el mismo modelo y escriben sus vectores en
  `base/Vectores` con el mismo esquema que la v2: la nube y el teléfono ven las
  mismas caras.

**Antes de hacer nada, lee en este orden:**

1. `AGENTS.md` — reglas que no se rompen (sobre todo: no tocar la v1, la base de
   KAXTEN solo con su llave, no desplegar, no copiar llaves, no commitear
   `config.py`, no hacer commits sin que te lo pida).
2. `docs/v2/ESTADO.md` — qué hay, qué está probado y qué sigue.
3. `docs/v2/API_V2.md` — endpoints, esquema de datos y variables de entorno.
4. `docs/v2/PLAN_V2.md` — fases A–E con su estado.
5. `docs/v2/DESPLIEGUE.md` y `docs/v2/PRUEBAS_LOCAL.md` — solo para entender cómo se
   despliega y cómo probar en local.

Si vas a tocar la integración con la app, lee también en `kaxten_base`:
`AGENTS.md`, `lib/services/api_facial_v2_service.dart`,
`lib/view_models/facial_embedding_viewmodel.dart`,
`lib/view_models/checkin_viewmodel.dart` y
`lib/services/vectores_local_cache_service.dart`.

**Cómo trabajar**

- Corre `python -m pytest -q` antes y después de cada cambio (no necesita
  TensorFlow ni Firebase; usa `requirements-dev.txt`).
- Recuerda que un push a `v2` que toque `app/`, `tests/`, `Dockerfile` o
  `requirements*.txt` **despliega a producción**. No hagas push sin que te lo pida.
- Nunca hagas altas ni bajas de prueba en el índice `desaparecidos` (es
  producción de KAXTEN). Para pruebas reales usa búsquedas o el índice `pruebas`.
- Nunca escribas una llave en código, documentación o un mensaje. Las llaves
  locales están en `.env.local` (ignorado); léelas con un script sin imprimirlas.
- Si algo no lo pudiste verificar, dilo. Si una decisión afecta datos de KAXTEN o
  la v1, pregúntame antes.

**Lo siguiente que quiero** (confírmame el plan antes de empezar):

[Escribe aquí la tarea. Opciones pendientes según `docs/v2/ESTADO.md`: fase B
(versiones fijas, pesos del modelo en la imagen, logs de tiempos), fase E4 (la app
de KAXTEN usando la v2) o mejoras pedidas para la demo.]

Primero lee los documentos y dime en pocas líneas qué entendiste del proyecto y
del estado actual, y si ves algún riesgo o algo que no cuadre.

---
