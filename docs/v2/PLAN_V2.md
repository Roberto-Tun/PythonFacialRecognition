# Plan de la v2 — API facial por índices

Rama: `v2` (sale de `cosine`, que es la que está en producción).
Antecedente: `kaxten_base/docs/referencia_kaxten/requerimientos_refactor_api_facial.md`
(problemas de la v1 y tareas T-01…T-17). Este plan toma de ahí lo que necesita la
demo del hackathon y lo separa en fases.

## Objetivo

Que un rostro dado de alta en un teléfono se pueda buscar en la nube, y al revés,
para cualquier cliente (KAXTEN Base hoy; KAXTEN u otro negocio después), **sin tocar
los datos de KAXTEN**.

## Decisiones tomadas (con Saul, 20/11/2026)

| Tema | Decisión |
|---|---|
| Rama | `v2` desde `cosine` |
| Dónde viven los vectores | Base `base`, colección `Vectores`, **el mismo esquema que ya escriben los teléfonos** |
| Cómo se busca | En memoria en el servidor (coseno con NumPy), por índice, con caché que se refresca. Para cientos o pocos miles de caras alcanza y no hay que crear nada con `gcloud`. La búsqueda vectorial de Firestore queda para cuando crezca (fase D) |
| Quién puede llamar | Una llave por cliente en la variable `V2_LLAVES`, que dice qué índices puede tocar. Saul **no** la configura todavía: en local se usa `V2_SIN_LLAVE=1` (solo para pruebas; dice en `/v2/salud` que está abierta) |
| Fotos | La v2 **no guarda fotos**. Solo vectores |
| v1 | Intacta. La v2 se monta a un lado (`/v2/...`) |
| Despliegue | Como un servicio **aparte** de Cloud Run (por ejemplo `kaxten-api-v2`), para no arriesgar el de KAXTEN. Lo hace Saul |

## Fase A — v2 mínima para la demo

| Id | Tarea | Listo cuando |
|---|---|---|
| A1 | Credenciales sin archivo: Firebase con las credenciales del entorno (ADC) y, solo si existe, el archivo de `config.FIREBASE_KEY` (para que Saul siga probando la v1 en local) | La imagen no contiene ningún `.json`; el servicio arranca con `GOOGLE_APPLICATION_CREDENTIALS` montado por volumen |
| A2 | `.dockerignore` y `Dockerfile` sin llave; `.pyc` fuera de git | `docker build` no copia credenciales ni cachés |
| A3 | `app/v2/config.py`: variables de entorno y la guarda que impide usar `(default)` | Con `V2_BASE_DATOS=(default)` la v2 no arranca |
| A4 | `app/v2/auth.py`: llave por cliente → índices permitidos | Sin llave → 401; índice ajeno → 403 |
| A5 | `app/v2/repositorio.py`: leer/escribir/borrar en `base/Vectores` con el esquema del teléfono; versión en memoria para pruebas | Un vector escrito por la v2 lo lee la app, y al revés |
| A6 | `app/v2/indice.py`: matriz por índice en memoria, refresco por tiempo (`V2_REFRESCO_S`) y al escribir | Una búsqueda no vuelve a leer toda la colección cada vez |
| A7 | `app/v2/vision.py`: detector + FaceNet con carga perezosa, **la cara más grande**, recorte con `max(0, x)` (T-04), orientación EXIF y foto reducida a ~1024 px | Una foto vertical del teléfono encuentra la cara |
| A8 | Endpoints: `salud`, `modelo`, `vectores` (foto → vector sin guardar), alta por foto y por vector, baja de un sujeto, búsqueda por foto y por vector | `tests/` en verde |
| A9 | Respuesta de búsqueda con umbral y límite, una fila por sujeto (su mejor vector), tiempos por etapa | Una cara que no está devuelve lista vacía, no "el más parecido al 30 %" |
| A10 | Docs: `API_V2.md`, `PRUEBAS_LOCAL.md` | Otro agente puede seguir sin preguntar |

## Fase B — seguridad y operación (del documento de requerimientos)

T-02 (fijar versiones con un `pip freeze` de una imagen que funcione), T-03
(`cloudbuild.yaml` con el servicio y la región reales), T-06 (tiempos en logs),
T-07 (pesos dentro de la imagen y calentamiento al arrancar), T-08 (concurrencia),
llaves en Secret Manager, límite de tamaño de las fotos, CORS si hiciera falta.

## Fase C — la app (`kaxten_base`)

- Cliente de la v2 (`AppConfig.usarServidorFacial` + URL + llave en `sistema.dart`).
  Búsqueda por **vector** (el teléfono ya lo calcula): más rápida y no sube fotos.
- Con red: búsqueda en la nube; sin red: la de hoy, en el teléfono.
- Descarga automática de vectores nuevos en los demás teléfonos (no necesita la
  v2: es escuchar `Vectores` en Firestore).

## Fase D — escala (solo si hace falta)

Búsqueda vectorial de Firestore (`find_nearest` con prefiltro por `indice`, índice
vectorial creado por Saul con `gcloud`), en lugar de la matriz en memoria.

## Riesgos conocidos

- **Detector distinto:** el servidor detecta con MTCNN y el teléfono con ML Kit. El
  recorte cambia un poco, y con él el vector. Por eso la app debe buscar **por
  vector** (mismo detector en alta y búsqueda). La búsqueda por foto sirve para
  clientes sin la app; su umbral se calibra aparte.
- **Primera petición lenta:** TensorFlow y los pesos se cargan con la primera foto
  (no con las búsquedas por vector, que no usan el modelo). Fase B lo resuelve.
- **Llaves abiertas en local:** `V2_SIN_LLAVE=1` es solo para la máquina de Saul.
  Nunca en Cloud Run.

## Estado

### Fase A — hecha y probada contra Firestore (20/11/2026)

- A1–A10 implementadas: `app/core/credenciales.py`, `app/v2/*`, `.dockerignore`,
  `Dockerfile` sin llave, `.pyc` fuera de git, `tests/` (23 pruebas en verde con
  repositorio en memoria y visión falsa).
- **Verificado en el contenedor con la llave montada** (índice `pruebas`, vectores
  sintéticos, borrados al final): alta (1.25 s la primera, por la conexión), búsqueda
  (0.30 s, similitud 0.9986), cara desconocida → vacío, grupo ajeno → vacío, baja →
  `borrados: 1`. Leyó el índice `miembros` que había escrito un teléfono (1 vector):
  el esquema compartido funciona.
- TensorFlow, MTCNN y FaceNet cargan en el contenedor (primera foto 14 s, siguientes
  al instante).
- **No verificado:** el vector de una **cara real** (solo se probó con una imagen
  sin cara), ni que un vector del servidor y uno del teléfono de la misma persona
  pasen el umbral. Es la primera prueba de la fase C.
- La v1 no cambió de comportamiento. Solo cambió de dónde salen las credenciales:
  si el archivo de `config.FIREBASE_KEY` existe se usa igual que antes; si no, ADC.

### Pendientes de Saul antes de desplegar

1. ~~Confirmar que no hay disparadores de Cloud Build~~: confirmado, no hay. `v2` ya
   está en GitHub.
2. Rotar la llave de cuenta de servicio que estuvo en la raíz del repo y sacarla de
   la carpeta (T-01).
   Además, el **historial** del repositorio tiene credenciales de otro proyecto
   (`facerecognitionapp-a1ca7-…json` y `keyFaceRecog.pem`): revocarlas en su consola.
   Borrarlas del historial exige reescribirlo (`git filter-repo`) y forzar el push.
3. Crear el servicio aparte (`kaxten-api-v2`) con `V2_SOLO=1`, `V2_LLAVES` desde
   Secret Manager y una cuenta de servicio con acceso a Firestore.

### Siguiente

Fase C (cliente en `kaxten_base`) en cuanto Saul confirme que la Fase A funciona
contra la base `base`.
