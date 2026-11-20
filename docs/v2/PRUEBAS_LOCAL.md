# Probar la v2 en local

## 1. Pruebas automáticas (sin Firebase ni TensorFlow, ~2 s)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
```

## 2. El contenedor

```powershell
docker build -t api-facial-v2:local .
```

### Sin credenciales (solo para ver que arranca)

```powershell
docker run --rm -p 8000:8000 -e V2_SOLO=1 -e V2_SIN_LLAVE=1 api-facial-v2:local
```

`http://localhost:8000/v2/salud` responde. Lo que toca Firestore contesta
`503 sin_credenciales`. Está bien así: no hay llave.

### Con credenciales, contra la base `base`

Hay dos caminos. Elige uno.

**A. Con la llave de cuenta de servicio, montada por volumen** (no entra a la imagen):

```powershell
$llave = "C:\ruta\fuera\del\repo\llave-de-servicio.json"
docker run --rm -p 8000:8000 `
  -e V2_SOLO=1 -e V2_SIN_LLAVE=1 `
  -v "${llave}:/secretos/llave.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/secretos/llave.json `
  api-facial-v2:local
```

**B. Con tu sesión de `gcloud`** (sin archivo de llave):

```powershell
# Una sola vez: instalar el SDK de Google Cloud
winget install --id Google.CloudSDK -e
# Cerrar y abrir la terminal, y luego:
gcloud auth login
gcloud config set project kaxten-2090b
gcloud auth application-default login

docker run --rm -p 8000:8000 `
  -e V2_SOLO=1 -e V2_SIN_LLAVE=1 `
  -v "$env:APPDATA\gcloud:/home/appuser/.config/gcloud:ro" `
  -e GOOGLE_CLOUD_PROJECT=kaxten-2090b `
  api-facial-v2:local
```

`V2_SOLO=1` deja fuera la v1: el contenedor no toca la base de KAXTEN.
`V2_SIN_LLAVE=1` abre todos los índices **sin llave**: úsalo solo en tu máquina.

### Probar a mano

Abre `http://localhost:8000/docs`: FastAPI arma una página para probar cada
endpoint desde el navegador. Un recorrido corto:

1. `POST /v2/indices/pruebas/sujetos/yo/rostros` con una foto tuya → `guardados: [...]`.
2. `POST /v2/indices/pruebas/busqueda/foto` con **otra** foto tuya → apareces con
   tu `similitud`.
3. La misma búsqueda con la foto de otra persona → `coincidencias: []`.
4. `DELETE /v2/indices/pruebas/sujetos/yo` → `borrados: 1`.
5. En la consola de Firebase, base `base` → `Vectores`: los documentos
   `pruebas_yo_s…` aparecen tras el paso 1 y desaparecen tras el paso 4.

Usa el índice `pruebas` para no mezclar con `miembros` y `registros`, que son los de
la app.

La **primera foto tarda** (decenas de segundos): ahí se cargan TensorFlow, MTCNN
y FaceNet, y keras-facenet baja sus pesos de internet. Las búsquedas por vector
no cargan el modelo. Meter los pesos en la imagen es la tarea T-07 (fase B).

### Con llaves de cliente, como en producción

```powershell
docker run --rm -p 8000:8000 -e V2_SOLO=1 `
  -e V2_LLAVES='{"<una llave larga y aleatoria>": {"cliente": "kaxten_base", "indices": ["miembros", "registros", "pruebas"]}}' `
  ... (credenciales como arriba) api-facial-v2:local
```

La llave se genera con
`python -c "import secrets; print(secrets.token_urlsafe(32))"` y nunca se escribe
en el repo ni en la documentación.
