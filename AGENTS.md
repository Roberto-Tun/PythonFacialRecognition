# AGENTS.md — API de reconocimiento facial (KAXTEN y KAXTEN Base)

Lee esto completo antes de tocar código. Después lee `docs/v2/PLAN_V2.md`.

## Qué es

API en FastAPI que convierte una cara en un vector de 512 números (FaceNet) y busca
caras parecidas. Hoy la usa **KAXTEN** (búsqueda de personas desaparecidas, en
producción) con la **v1**: `/register_missing_person/` e `/identify/`.

La **v2** (rama `v2`) es un servicio genérico por **índices**: un índice es un
conjunto de caras con nombre (`miembros` para el check-in de un gimnasio,
`registros` para los Registros de la plantilla del hackathon…). La usa **KAXTEN
Base** (`C:\Users\Saul Colli\Documents\kaxten_base`, la plantilla genérica en
Flutter). Más adelante KAXTEN podría pasarse a la v2 con su propio índice.

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
   la normalización de `keras_facenet`. Si cambia cualquiera de las tres, los
   vectores del servidor y los del teléfono dejan de ser comparables.
5. **No desplegar.** Ni `gcloud run deploy`, ni `gcloud builds submit`, ni lanzar a
   mano el flujo `.github/workflows/desplegar-v2.yml`, ni tocar la variable
   `DESPLEGAR_V2` del repositorio. Ojo: con esa variable en `true`, **cada push a
   `v2` que toque `app/`, `tests/`, `Dockerfile` o `requirements` despliega** a Cloud
   Run. Ver `docs/v2/DESPLIEGUE.md`.
6. **No copiar llaves.** Ningún `.json` de cuenta de servicio, `.pem` ni llave de
   cliente en el código, la imagen, la documentación o un mensaje. Solo nombres de
   variables de entorno.
7. **`config.py` tiene cambios locales de Saul.** No se commitea nunca. Si hace falta
   un ajuste, va por variable de entorno.
8. **No hacer commits ni push** salvo que Saul lo pida.

## Forma de trabajar

- Fase por fase (`docs/plantilla` no aplica aquí: el plan está en
  `docs/v2/PLAN_V2.md`). Al cerrar cada fase: pruebas en verde y un resumen para
  Saul de qué cambió y qué probar.
- Español en comentarios, nombres de dominio y mensajes de error de la v2. Los
  comentarios explican **por qué**.
- Lo que no se pudo verificar (un despliegue, una base real) se dice
  explícitamente.

## Estructura

```
app/main.py            monta la v1 y la v2
app/api/routes.py      v1 (KAXTEN) — no tocar salvo correcciones acordadas
app/core/              v1: detector, embedder, Firebase
app/v2/                v2: config, auth, esquemas, repositorio, índice en memoria,
                       visión (detector + FaceNet, carga perezosa), rutas
tests/                 pruebas de la v2 (sin TensorFlow ni Firebase)
docs/v2/               plan, referencia de la API y cómo probar en local
```
