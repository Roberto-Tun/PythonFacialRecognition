# Estado de la API facial (21/11/2026)

Para quien toma el proyecto. Qué existe, qué está probado y qué falta.

## Para qué sirve

Un servicio de **reconocimiento facial por índices**: un cliente da de alta caras
(como vectores de 512 números) en un índice y luego busca con una foto o un
vector quién es. Lo usan:

| Cliente | Índices | Base de Firestore | Para qué |
|---|---|---|---|
| **KAXTEN Base** (app Flutter, plantilla del hackathon) | `miembros`, `registros` | `base` | Check-in facial de gimnasio/restaurante y búsqueda de Registros por rostro |
| **KAXTEN** (app publicada de personas desaparecidas) | `desaparecidos` | `(default)` | Buscar a una persona entre los reportes. Hoy su app usa la **v1**; la v2 ya puede atenderla (fase E) |

Idea central: **el teléfono también reconoce** (mismo FaceNet en TFLite, sin red) y
escribe sus vectores en `base/Vectores` con el mismo esquema que la v2. La nube
no reemplaza al teléfono: completa lo que el teléfono no tiene.

## Dónde está

| | |
|---|---|
| Código | Repo `Roberto-Tun/PythonFacialRecognition`, rama **`v2`** (sale de `cosine`, que es la v1 en producción) |
| Producción | Cloud Run `kaxten-api-v2`, `northamerica-south1`, proyecto `kaxten-2090b`: `https://kaxten-api-v2-686865780124.northamerica-south1.run.app` (`/docs` para probar) |
| Despliegue | GitHub Actions (`.github/workflows/desplegar-v2.yml`): cada push a `v2` que toque el código → pruebas → Artifact Registry (`api-facial`) → Cloud Run. Variable `DESPLEGAR_V2=true` |
| Configuración en la nube | `V2_SOLO=1` (sin v1) y `V2_LLAVES` desde Secret Manager (`v2-llaves`) |
| Imágenes en Docker Hub | `saulcolli/api-facial-v2:2.0.0-a` (sin llave). `saulcolli/api-facial-v2-privada` tiene una llave de servicio adentro: debe ser privada y la llave rotarse |
| La app | `C:\Users\Saul Colli\Documents\kaxten_base`, `lib/services/api_facial_v2_service.dart`. Ya apunta a Cloud Run con su llave (en `lib/sistema.dart`, fuera de git) |

## Qué está hecho

- **Fase A — v2 mínima:** índices, alta (vector y foto), baja, búsqueda (vector y
  foto), salud, modelo; llaves por cliente; esquema compartido con los teléfonos;
  correcciones T-04 (cara más grande, recorte dentro de la foto, giro EXIF, foto
  ≤ 1024 px) y T-01 (sin llave en la imagen).
- **Fase C — la app:** búsqueda facial teléfono + nube (por vector, sin subir
  fotos); check-in con nube solo cuando el teléfono duda; vectores nuevos de otros
  teléfonos llegan solos; se borró de la app todo camino a la v1.
- **Fase E (E1–E3) — KAXTEN en la v2:** índice `desaparecidos` sobre sus datos sin
  migrarlos, solo con su llave, con filtro de visibilidad (T-05), sin crear
  personas y con confirmación para borrar.
- **Comodidad:** `POST /v2/busqueda/foto` sin índice (busca en todos los de la
  llave); valores por defecto (umbral 0.5, límite 10); cada coincidencia trae
  `datos` del sujeto y `foto_url`; `/docs` con selector de archivos y botón
  Authorize.

## Qué está probado

| Prueba | Resultado |
|---|---|
| `python -m pytest -q` | 42 en verde (sin TensorFlow ni Firebase) |
| Contenedor local con la base `base` | Alta, búsqueda (0.30 s), baja, grupo ajeno vacío |
| **Cloud Run, llave de la plantilla** (21/11) | Foto sin índice → socio "Saulo" 0.8156 (alta hecha **en el teléfono**, encontrada **por el servidor**); la llamada exacta de la app (por vector) en 0.26 s; 403 al intentar KAXTEN |
| **Cloud Run, llave de KAXTEN** (21/11, solo lectura) | 4,742 rostros; 8 coincidencias con umbral 0.7, con datos y foto; 403 al intentar `miembros` |
| GitHub Actions | Pruebas y despliegue en verde |

## Qué no está probado

- Altas y bajas reales en `desaparecidos` (a propósito: es producción).
- La app de la plantilla contra la nube **en el teléfono** (lo prueba Saul; lista en
  `kaxten_base/docs/plantilla/06_pendientes_saul.md`, sección "API v2 — fase C").

## Qué sigue (en orden)

1. **Antes de la demo:** `CLOUD_RUN_MIN_INSTANCIAS=1` en GitHub (evita ~20 s en la
   primera foto tras un arranque en frío).
2. **Seguridad:** rotar las llaves de clientes de `v2-llaves` (se compartieron en un
   chat), confirmar que `api-facial-v2-privada` es privada y rotar la llave de
   servicio que lleva adentro, borrar la llave vieja de `github-despliegue`.
3. **Fase B (operación):** fijar versiones en `requirements.txt` (`pip freeze` de la
   imagen que funciona), pesos del modelo dentro de la imagen (T-07), tiempos en
   logs (T-06), concurrencia (T-08), actualizar las acciones de GitHub que avisan
   de Node.js 20.
4. **Fase E (E4–E5):** que la app de KAXTEN busque por vector en la v2 y, cuando
   ninguna versión publicada use la v1, retirarla.
5. **Fase D (escala):** si un índice pasa de decenas de miles de caras, cambiar la
   matriz en memoria por la búsqueda vectorial de Firestore.

## Números útiles

- Primera foto tras arrancar: ~15–22 s (carga TensorFlow + pesos). Siguientes:
  ~0.5 s de visión.
- Búsqueda por vector: ~0.3 s (lo que usa la app).
- `desaparecidos`: la primera carga de 4,742 rostros tarda ~7 s; luego se reusa
  10 min (`V2_REFRESCO_KAXTEN_S`).
- Similitud de la misma persona teléfono ↔ servidor: 0.82 (umbral del check-in 0.65).
