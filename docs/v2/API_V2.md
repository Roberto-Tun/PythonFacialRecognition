# Referencia de la v2

Base: `https://<servicio>/v2`. Todo es JSON salvo las fotos (`multipart/form-data`,
campo `fotos`, de 1 a 5, máximo 8 MB cada una).

## Autenticación

Encabezado `X-Api-Key: <llave del cliente>`. Cada llave dice qué índices puede tocar
(variable `V2_LLAVES`, ver abajo). `GET /v2/salud` y `GET /v2/modelo` no la piden.

| Estado | `error` | Cuándo |
|---|---|---|
| 401 | `llave_invalida` | Falta la llave o no existe |
| 403 | `indice_ajeno` | La llave no tiene ese índice |
| 503 | `sin_configurar` | El servidor no tiene `V2_LLAVES` ni `V2_SIN_LLAVE=1` |
| 503 | `sin_credenciales` | El servidor no puede entrar a Firestore |

Todos los errores tienen la forma `{"error": "<código>", "mensaje": "<texto en español>"}`.
Nunca traen el texto de la excepción interna.

## Conceptos

- **Índice**: un conjunto de caras (`miembros`, `registros`…). Minúsculas, números y `_`.
- **Sujeto**: a quién pertenece la cara (el id del socio, del Registro…). Un sujeto
  puede tener varios vectores (varias fotos).
- **Grupo**: subconjunto opcional dentro del índice. La app lo usa para separar
  negocios del check-in (`demo_gimnasio`, `demo_restaurante`).
- **Modelo**: `facenet_512` (servidor) y `facenet_512_fp16` (teléfono) son
  compatibles. Otro valor → `409 modelo_incompatible`.

## Endpoints

### `GET /v2/salud`
`{"estado": "ok", "version": "...", "modelo": "facenet_512", "base_datos": "base", "autenticacion": "llaves"}`

### `GET /v2/modelo`
Modelo, modelos compatibles, dimensión (512), recorte y comparación (coseno).

### `POST /v2/vectores` — foto → vector, sin guardar
Respuesta: `{"resultados": [{"foto": 0, "modelo": "...", "vector": [...512], "caja": [x,y,w,h], "confianza": 0.99, "caras_detectadas": 1}]}`.
Una foto sin cara trae `{"foto": i, "error": "sin_rostro"}`.

### `POST /v2/indices/{indice}/sujetos/{sujeto_id}/vectores` — alta con vectores (la app)
```json
{ "modelo": "facenet_512_fp16", "vectores": [[...512]], "grupo": "demo_gimnasio" }
```
Respuesta: `{"indice": "...", "sujeto_id": "...", "guardados": ["<id>", ...], "sin_rostro": [], "caras_ignoradas": 0}`.
Es idempotente: el id del documento sale del vector, así que repetir la petición no
duplica nada.

### `POST /v2/indices/{indice}/sujetos/{sujeto_id}/rostros` — alta con fotos
`multipart`: `fotos` (1–5) y `grupo` (opcional). Guarda **la cara más grande** de
cada foto y dice cuántas ignoró. Las fotos no se guardan. Si ninguna foto tiene
cara: `422 sin_rostro`.

### `DELETE /v2/indices/{indice}/sujetos/{sujeto_id}`
Borra todos los vectores del sujeto en ese índice, **también los que subió un
teléfono**. `{"borrados": n}`.

### `POST /v2/indices/{indice}/busqueda/vector` — búsqueda (la app)
```json
{ "modelo": "facenet_512_fp16", "vectores": [[...512]], "limite": 10, "umbral": 0.5, "grupo": "demo_gimnasio" }
```
`limite` (por defecto y máximo `V2_LIMITE_MAX`, 50), `umbral` (0–1, por defecto
`V2_UMBRAL`, 0.5) y `grupo` son opcionales.

### `POST /v2/indices/{indice}/busqueda/foto`
`multipart`: `fotos`, y opcionales `limite`, `umbral` y `grupo`.

### Respuesta de las dos búsquedas
```json
{
  "indice": "miembros",
  "coincidencias": [
    { "sujeto_id": "abc", "grupo": "demo_gimnasio", "similitud": 0.8731, "id_vector": "miembros_abc_0" }
  ],
  "comparados": 42,
  "umbral": 0.5,
  "limite": 10,
  "sin_rostro": [],
  "tiempos_ms": { "vision": 812.4, "busqueda": 0.6, "total": 815.0 }
}
```
- Una fila **por sujeto** (su vector más parecido), ordenadas de mayor a menor.
- Solo las que pasan el umbral: una cara desconocida da `coincidencias: []`. La v1
  siempre devolvía "al más parecido", aunque fuera al 30 %.
- `comparados`: cuántos vectores había en el índice (y grupo) al buscar.

## Datos: `base` → `Vectores`

El mismo esquema que escriben los teléfonos, así que la nube y la app ven las
mismas caras:

| Campo | Tipo | |
|---|---|---|
| `indice` | texto | |
| `sujeto_id` | texto | |
| `grupo` | texto | `""` si no hay |
| `vector` | arreglo de 512 números | |
| `modelo` | texto | `facenet_512` o `facenet_512_fp16` |
| `fecha` | texto ISO 8601 | |

Ids: `<indice>_<sujeto>_<n>` los del teléfono y `<indice>_<sujeto>_s<huella>` los del
servidor.

El servidor tarda hasta `V2_REFRESCO_S` (30 s) en ver un alta que un teléfono
escribió directo en Firestore. Las altas hechas por la v2 se ven al instante.

## Variables de entorno

| Variable | Por defecto | |
|---|---|---|
| `V2_BASE_DATOS` | `base` | `(default)` se rechaza al arrancar: es la de KAXTEN |
| `V2_COLECCION` | `Vectores` | |
| `V2_LLAVES` | vacío | JSON `{"<llave de 24+ caracteres>": {"cliente": "kaxten_base", "indices": ["miembros", "registros"]}}`. En Cloud Run, desde Secret Manager |
| `V2_SIN_LLAVE` | vacío | `1` = sin llave, todos los índices. **Solo en local** |
| `V2_SOLO` | vacío | `1` = no se monta la v1 (no toca la base de KAXTEN) |
| `V2_REFRESCO_S` | `30` | Cada cuánto se releen los índices |
| `V2_LIMITE_MAX` | `50` | |
| `V2_UMBRAL` | `0.5` | Umbral por defecto. La app manda el suyo (`umbralCheckIn`, `umbralBusquedaFacial`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | vacío | Solo en local, apuntando a la llave **montada** |
