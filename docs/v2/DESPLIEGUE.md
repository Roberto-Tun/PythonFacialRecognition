# Desplegar la v2 en Cloud Run

Dos caminos. El **A** es para tenerla arriba hoy en 5 minutos; el **B** deja el
despliegue automático con GitHub Actions en cada push a `v2`.

Todo se hace desde **Cloud Shell** (consola de Google Cloud → icono `>_` arriba a
la derecha): ya trae `gcloud` y la sesión iniciada.

## Antes: la llave no va en la imagen

En Cloud Run **no hace falta** meter la llave de servicio en la imagen: el
servicio corre con una cuenta de servicio del proyecto y la v2 toma sus
credenciales sola (`app/core/credenciales.py`, ADC). Meterla no agrega nada y la
deja en cada copia de la imagen. Por eso el flujo de GitHub construye la imagen
**sin llave**, y la sube a Artifact Registry, que es privado.

Las "variables de entorno en la nube" son solo dos:

| Variable | Dónde | Valor |
|---|---|---|
| `V2_SOLO` | variable normal | `1` (no se monta la v1) |
| `V2_LLAVES` | **Secret Manager**, secreto `v2-llaves` | el mismo JSON de `.env.local`, con llaves **nuevas** (no las de prueba) |

## Una sola vez: preparar el proyecto (Cloud Shell)

```bash
PROYECTO=kaxten-2090b
REGION=northamerica-south1        # la del servicio actual de KAXTEN; cámbiala si es otra
gcloud config set project $PROYECTO
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# Repositorio privado de imágenes
gcloud artifacts repositories create api-facial --repository-format=docker --location=$REGION

# Llaves de los clientes (genera llaves nuevas; no reutilices las de .env.local)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # una por cliente
nano llaves.json
#   {"<llave demo>": {"cliente": "kaxten_base", "indices": ["miembros", "registros"]},
#    "<llave kaxten>": {"cliente": "kaxten", "base_datos": "(default)", "indices": ["desaparecidos"]}}
gcloud secrets create v2-llaves --data-file=llaves.json
rm llaves.json

# La cuenta con la que corre Cloud Run puede leer el secreto
NUM=$(gcloud projects describe $PROYECTO --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding v2-llaves \
  --member=serviceAccount:$NUM-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

La cuenta `…-compute@developer.gserviceaccount.com` es la que usa Cloud Run por
defecto; normalmente tiene acceso a Firestore. Si al buscar sale
`sin_credenciales` o un 403 de Firestore, dale el rol `roles/datastore.user`.

## A. Desplegar ya, desde la imagen de Docker Hub

La imagen **pública y sin llave** sirve tal cual:

```bash
gcloud run deploy kaxten-api-v2 \
  --image docker.io/saulcolli/api-facial-v2:2.0.0-a \
  --region $REGION --port 8000 --memory 2Gi --cpu 2 --timeout 300 \
  --allow-unauthenticated \
  --set-env-vars V2_SOLO=1 \
  --set-secrets V2_LLAVES=v2-llaves:latest
```

Al terminar imprime la URL. `https://<url>/v2/salud` debe decir
`"autenticacion": "llaves"`, y `https://<url>/docs` abre la página de pruebas.

Cloud Run **no puede** bajar imágenes privadas de Docker Hub sin configuración
extra. La privada (`api-facial-v2-privada`) solo sirve para correrla en máquinas
del equipo.

## B. Despliegue automático con GitHub Actions

El flujo está en `.github/workflows/desplegar-v2.yml`: en cada push a `v2` que toque
el código corre las pruebas y, si pasan, construye la imagen, la sube a Artifact
Registry y despliega `kaxten-api-v2`.

### 1. Una cuenta de servicio para GitHub (Cloud Shell)

```bash
SA=github-despliegue@$PROYECTO.iam.gserviceaccount.com
gcloud iam service-accounts create github-despliegue --display-name="GitHub Actions: API v2"
for ROL in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROYECTO --member=serviceAccount:$SA --role=$ROL
done
gcloud iam service-accounts keys create github-despliegue.json --iam-account=$SA
cat github-despliegue.json      # copia TODO el contenido para el paso 2
rm github-despliegue.json
```

Esta cuenta solo puede desplegar; **no** es la de Firebase Admin.

### 2. En GitHub

Repositorio → **Settings → Secrets and variables → Actions** (hace falta ser
administrador del repo):

- **Secrets** → *New repository secret*: `GCP_SA_KEY` = el JSON del paso anterior.
- **Variables** → *New repository variable*:
  - `GCP_PROJECT` = `kaxten-2090b`
  - `GCP_REGION` = la región que usaste arriba
  - `DESPLEGAR_V2` = `true` ← sin esta, el flujo solo corre las pruebas
  - opcional `CLOUD_RUN_MIN_INSTANCIAS` = `1` para la demo (sin arranques en frío; cuesta mientras esté encendido)

### 3. Lanzarlo

- **Actions → "API v2 — pruebas y despliegue" → Run workflow** (rama `v2`), o
- cualquier push a `v2` que toque `app/`, `tests/`, `Dockerfile` o `requirements`.

El último paso del flujo imprime la URL y consulta `/v2/salud`.

## Cambiar las llaves de los clientes

```bash
nano llaves.json && gcloud secrets versions add v2-llaves --data-file=llaves.json && rm llaves.json
gcloud run services update kaxten-api-v2 --region $REGION --set-secrets V2_LLAVES=v2-llaves:latest
```

## En la app (`kaxten_base`)

`DOMINIO_GLOBAL` = la URL del servicio con `/` al final, `SistemaApiFacial.LLAVE_V2` =
la llave del cliente `kaxten_base`, y `AppConfig.usarServidorFacial = true`.
