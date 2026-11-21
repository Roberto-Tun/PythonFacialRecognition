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

## Fase E — KAXTEN en la v2 (migración a futuro)

> **Estado (21/11/2026): E1, E2 y E3 hechas en código**, con 12 pruebas en
> `tests/test_kaxten.py` (repositorio en memoria). Se eligió el **adaptador** de
> lectura (`app/v2/kaxten.py`): no se migró ni se tocó ningún documento.
> **No verificado contra la base `(default)` real**: el agente no hizo pruebas
> contra producción. Lo prueba Saul con una llave de KAXTEN (`docs/v2/PRUEBAS_LOCAL.md`).
> E4 (la app de KAXTEN usando la v2) y E5 (retirar la v1) siguen pendientes.

La v1 **no se deshabilita**: la app publicada de KAXTEN la sigue usando. Esta fase
prepara que KAXTEN pueda pasarse a la v2 cuando se decida.

| Id | Tarea | Listo cuando |
|---|---|---|
| E1 | **Base por cliente.** Hoy la v2 rechaza `(default)` para proteger a KAXTEN. En `V2_LLAVES`, cada cliente declara su base (`"base_datos": "base"` para la demo, `"(default)"` para KAXTEN) y el repositorio se elige por cliente. La guarda pasa a ser: solo el cliente `kaxten` puede usar `(default)`, y solo con su llave | Una llave de la demo nunca lee ni escribe en `(default)`, con prueba automática |
| E2 | **Índice de KAXTEN** (`desaparecidos`) sobre los documentos de `Vectores` de `(default)`, que hoy no tienen `indice` y guardan `id_persona_desaparecida` como referencia. Opciones: un adaptador de lectura que trate ese esquema viejo como el índice `desaparecidos` (sin tocar los datos), o una migración que agregue `indice`/`sujeto_id` a cada documento. Recomendado: el adaptador, porque no escribe en producción | `/v2/indices/desaparecidos/busqueda/vector` encuentra lo mismo que `/identify/` |
| E3 | **Filtro de visibilidad** (T-05): un reporte oculto no debe salir como coincidencia. En la v2 genérica no aplica; en el índice `desaparecidos` sí | Un reporte con `visibilidad` no pública no aparece |
| E4 | **La app de KAXTEN** busca por vector en la v2, con la v1 como respaldo mientras dure la migración | Las dos versiones de la app conviven |
| E5 | **Retirar la v1** cuando ninguna versión publicada la use | Métricas de Cloud Run sin llamadas a `/identify/` en N semanas |

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

### Fase C — hecha (20/11/2026) y probada contra Cloud Run (21/11/2026)

> Con la llave de la plantilla, la llamada exacta de la app (búsqueda por vector en
> `miembros`) encontró en 0.26 s al socio que se dio de alta **en el teléfono**
> (0.8156). Falta la prueba en el teléfono con la nube encendida.

En `kaxten_base` (detalle en su `docs/plantilla/03_plan_de_trabajo.md`, sección
"Después del plan"):

- `lib/services/api_facial_v2_service.dart`: búsqueda por vector, tope de 5 s,
  motivo explícito cuando no usa la nube. Se enciende con
  `AppConfig.usarServidorFacial`, `DOMINIO_GLOBAL` (URL de la v2) y
  `SistemaApiFacial.LLAVE_V2` en `lib/sistema.dart`.
- Búsqueda facial: teléfono + nube, juntadas por Registro. Check-in: la nube solo
  cuando el teléfono duda.
- Vectores nuevos de otros teléfonos llegan solos (`escucharCambios`).
- Se borró de la app todo el camino a la v1.
- Verificado: `test/api_facial_v2_test.dart` contra este contenedor (alta en la v2,
  búsqueda desde el cliente de la app, similitud > 0.99, baja).
- **No verificado:** en teléfono, ni con una cara real (MTCNN del servidor contra
  ML Kit del teléfono).

### Siguiente

Fase B (operación) antes de la demo, sobre todo T-07 (pesos en la imagen) o
`--min-instances 1`. Fase E cuando se decida migrar KAXTEN.
