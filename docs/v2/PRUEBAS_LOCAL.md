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

En **PowerShell**, no en Git Bash: Git Bash convierte `/secretos/llave.json` en
`C:/Program Files/Git/secretos/...` y el contenedor no encuentra la llave.

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
endpoint desde el navegador. Lo más rápido: `POST /v2/busqueda/foto` → "Try it
out" → elige la foto → "Execute". Sin índice busca en todos. La llave va en el
botón **Authorize** (en modo abierto no hace falta). Un recorrido completo:

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

## 3. Imagen solo local con la llave adentro (`api-facial-v2:local-con-llave`)

Por prisa, para no montar la llave cada vez. **Nunca se sube a ningún registro**
ni se usa para desplegar: tiene la llave de servicio adentro. Se arma a partir de
la imagen normal, sin tocar el `Dockerfile` del repo:

```powershell
docker build -t api-facial-v2:local .
$llave = "C:\Users\Saul Colli\Documents\PythonFacialRecognition\kaxten-2090b-firebase-adminsdk-fbsvc-bad426e2b5.json"
docker rm -f armado 2>$null
docker create --name armado api-facial-v2:local
docker cp $llave armado:/app/llave-local.json
docker commit --change "ENV GOOGLE_APPLICATION_CREDENTIALS=/app/llave-local.json" armado api-facial-v2:local-con-llave
docker rm armado
```

Y se corre sin `-v`:

```powershell
docker run --rm -p 8000:8000 -e V2_SOLO=1 -e V2_SIN_LLAVE=1 api-facial-v2:local-con-llave
```

Para borrarla: `docker rmi api-facial-v2:local-con-llave`.

## 4. Probar el índice de KAXTEN (`desaparecidos`)

⚠️ **Es la base de producción de KAXTEN.** La búsqueda solo lee. Una alta
**escribe** en `Vectores` de KAXTEN (y por foto, en `rostros/`), y la app de
KAXTEN la verá. Para probar, usa solo la búsqueda, o una persona de prueba que
ya exista y bórrala después con `?confirmar=true`.

`V2_SIN_LLAVE` no abre este índice: hace falta una llave de KAXTEN.

```powershell
$k = python -c "import secrets; print(secrets.token_urlsafe(32))"
$d = python -c "import secrets; print(secrets.token_urlsafe(32))"
$llaves = '{"' + $k + '": {"cliente": "kaxten", "base_datos": "(default)", "indices": ["desaparecidos"]}, "' + $d + '": {"cliente": "kaxten_base", "indices": ["miembros", "registros", "pruebas"]}}'
docker run --rm -p 8000:8000 -e V2_SOLO=1 -e "V2_LLAVES=$llaves" api-facial-v2:local-con-llave
"Llave KAXTEN: $k"
```

En `http://localhost:8000/docs`, en `POST /v2/indices/desaparecidos/busqueda/foto`
pon la llave de KAXTEN en el campo `X-Api-Key`, y sube la foto de una
persona registrada en KAXTEN. Debe aparecer su id (el mismo que devolvería
`/identify/`), salvo que su reporte esté desactivado o retirado.

## 5. Llaves locales en `.env.local`

`PythonFacialRecognition\.env.local` (git y Docker lo ignoran) tiene dos llaves de
prueba generadas al azar, en sus comentarios: una de KAXTEN (`desaparecidos`) y una
de la demo (`miembros`, `registros`, `pruebas`). El contenedor se corre así:

```powershell
docker run -d --name v2local -p 8000:8000 --env-file .env.local api-facial-v2:local-con-llave
```

En `/docs`, botón **Authorize** → pega una de las dos. Con la de KAXTEN,
`POST /v2/busqueda/foto` busca entre las personas desaparecidas.
