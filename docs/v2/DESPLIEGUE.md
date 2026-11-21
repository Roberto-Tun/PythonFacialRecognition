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
REGION=northamerica-south1        # la del servicio desplegado (21/11/2026)
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

> Hecho el 21/11/2026: https://kaxten-api-v2-686865780124.northamerica-south1.run.app
> (`/docs` para probar).

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

> ✅ Configurado el 21/11/2026 (`DESPLEGAR_V2=true`): cada push a `v2` que toque el
> código despliega a producción.

### Qué es y qué hace aquí

GitHub Actions es una computadora prestada de GitHub que se enciende sola cuando
pasa algo en el repositorio (aquí: un push a la rama `v2`) y sigue una receta.
La receta es `.github/workflows/desplegar-v2.yml` y tiene dos partes:

1. **`pruebas`**: instala Python y corre `pytest`. Si algo falla, se detiene y no
   despliega nada roto.
2. **`desplegar`** (solo si las pruebas pasan **y** existe la variable
   `DESPLEGAR_V2=true`):
   1. entra a Google Cloud con la llave de una cuenta de servicio (`GCP_SA_KEY`);
   2. construye la imagen con el `Dockerfile` (sin llave adentro);
   3. la sube a **Artifact Registry** (`api-facial`, privado, en tu proyecto);
   4. actualiza el servicio **`kaxten-api-v2`**, el mismo que creaste con la opción
      A: misma URL, nueva versión;
   5. consulta `/v2/salud` para confirmar que responde.

Es lo mismo que hiciste a mano en la opción A, pero automático en cada cambio.

```
push a v2 ──► pruebas ──► ¿DESPLEGAR_V2=true? ──► construir ──► Artifact Registry ──► Cloud Run
                 │ falla                 │ no
                 └──► se detiene         └──► solo pruebas
```

### Lo que ya tienes (de la opción A)

- Proyecto `kaxten-2090b`, región `northamerica-south1`.
- El servicio `kaxten-api-v2` corriendo.
- El secreto `v2-llaves` y el permiso para leerlo.

Falta: el repositorio de imágenes (si no lo creaste), una cuenta de servicio para
GitHub, y darle a GitHub esos datos.

### Paso 1 — En Cloud Shell (5 min)

Abre Cloud Shell (icono `>_` arriba a la derecha en console.cloud.google.com) y
pega esto **tal cual**:

```bash
PROYECTO=kaxten-2090b
REGION=northamerica-south1
gcloud config set project $PROYECTO

# 1. Repositorio privado de imágenes (si ya existe, dirá "already exists": está bien)
gcloud artifacts repositories create api-facial \
  --repository-format=docker --location=$REGION

# 2. Cuenta de servicio solo para que GitHub despliegue
gcloud iam service-accounts create github-despliegue \
  --display-name="GitHub Actions: API v2"
SA=github-despliegue@$PROYECTO.iam.gserviceaccount.com

# Google tarda unos segundos en "ver" la cuenta nueva. Sin esta espera, los
# comandos siguientes fallan con "Service account ... does not exist".
sleep 30
gcloud iam service-accounts describe $SA --format='value(email)'   # debe imprimir el correo

# 3. Permisos mínimos: desplegar en Cloud Run, subir imágenes y
#    "actuar como" la cuenta con la que corre el servicio
for ROL in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROYECTO \
    --member=serviceAccount:$SA --role=$ROL --condition=None --format=none
done
# Comprobar: deben salir los tres roles
gcloud projects get-iam-policy $PROYECTO --flatten=bindings \
  --filter="bindings.members:$SA" --format="value(bindings.role)"

# 4. Su llave (un JSON). Se muestra en pantalla para copiarla.
gcloud iam service-accounts keys create github-despliegue.json --iam-account=$SA
cat github-despliegue.json
```

Copia **todo** lo que imprimió el último comando, desde `{` hasta `}`, y luego
bórralo de Cloud Shell:

```bash
rm github-despliegue.json
```

> Esta cuenta solo puede desplegar y subir imágenes: no puede leer Firestore. Si
> esa llave se filtrara, el daño sería mucho menor que con la de Firebase Admin.
>
> Si algún comando dice **"Service account ... does not exist"** justo después de
> crear la cuenta, no pasó nada malo: es la demora de Google. Espera un minuto y
> repite desde el paso 3 (repetir los permisos no hace daño). Visto el 21/11/2026.
>
> Si el paso 4 falla con `constraints/iam.disableServiceAccountKeyCreation`, tu
> organización prohíbe llaves de servicio: hay que cambiar a "Workload Identity
> Federation" (sin llaves). Pídeselo al agente.

### Paso 2 — En GitHub (3 min)

Necesitas ser **administrador** del repositorio
(`github.com/Roberto-Tun/PythonFacialRecognition`). Si no ves la pestaña
*Settings*, pídele a Roberto que te dé el rol *Admin* o que haga este paso.

1. Repositorio → **Settings** → menú izquierdo **Secrets and variables** →
   **Actions**.
2. Pestaña **Secrets** → **New repository secret**:
   - *Name*: `GCP_SA_KEY`
   - *Secret*: pega el JSON del paso 1 completo.
   - **Add secret**.
3. Pestaña **Variables** → **New repository variable**, una por una:

   | Name | Value | Para qué |
   |---|---|---|
   | `GCP_PROJECT` | `kaxten-2090b` | El proyecto |
   | `GCP_REGION` | `northamerica-south1` | Dónde vive el servicio |
   | `DESPLEGAR_V2` | `true` | El interruptor: sin esto solo corren las pruebas |
   | `CLOUD_RUN_MIN_INSTANCIAS` | `1` | *(Opcional, para la demo)* una instancia siempre encendida: sin arranques en frío de 15 s. Cuesta mientras esté encendida; vuelve a `0` después del hackathon |

Los *Secrets* nadie los puede volver a leer (ni tú): solo los usa el flujo. Las
*Variables* sí se ven; por eso ahí no va nada sensible.

### Paso 3 — Lanzarlo

GitHub muestra el botón **Run workflow** solo para flujos que estén en la rama
por defecto (`main`). Mientras este flujo viva únicamente en `v2`, se lanza con
un **push a `v2` que toque** `app/`, `tests/`, `Dockerfile`, `requirements*.txt`
o el propio flujo. Un commit vacío **no** lo dispara (por ese filtro).

Para la primera vez, lo más simple: abre `.github/workflows/desplegar-v2.yml`,
agrega una línea de comentario al inicio (por ejemplo
`# Primer despliegue automático`), guarda y:

```powershell
cd "C:\Users\Saul Colli\Documents\PythonFacialRecognition"
git checkout v2
git add .github/workflows/desplegar-v2.yml
git commit -m "ci: primer despliegue con GitHub Actions"
git push
```

Si más adelante `v2` se une a `main`, aparecerá **Actions → API v2 — pruebas y
despliegue → Run workflow** para lanzarlo sin cambios.

### Paso 4 — Ver qué pasó

Repositorio → pestaña **Actions** → la ejecución más reciente:

- Círculo amarillo: corriendo (unos 8–12 min; casi todo es construir la imagen
  con TensorFlow).
- ✅ Verde: listo. Abre el paso **"Comprobar que responde"**: ahí están la URL y la
  respuesta de `/v2/salud`.
- ❌ Rojo: entra al paso que falló y lee el error. Los más comunes:

| Error | Causa | Arreglo |
|---|---|---|
| El job `desplegar` sale como *skipped* | Falta `DESPLEGAR_V2=true` | Paso 2.3 |
| `credentials_json` vacío o `failed to parse` | El secreto `GCP_SA_KEY` no está o se copió incompleto | Repetir pasos 1.4 y 2.2 |
| `Repository "api-facial" not found` | No existe el repositorio de imágenes, o está en otra región | Paso 1.1, con la misma región que `GCP_REGION` |
| `Permission 'artifactregistry.repositories.uploadArtifacts' denied` | Falta el rol de Artifact Registry | Paso 1.3 |
| `Permission 'iam.serviceaccounts.actAs' denied` | Falta `roles/iam.serviceAccountUser` | Paso 1.3 |
| `Secret v2-llaves ... permission denied` al arrancar | La cuenta de Cloud Run no puede leer el secreto | Sección "Una sola vez: preparar el proyecto", último comando |
| Fallan las `pruebas` | Un cambio rompió algo | Corre `python -m pytest -q` en tu máquina antes del push |

### Desde entonces

Cada push a `v2` que toque el código despliega solo, a la misma URL. Para
**apagar** el despliegue automático sin borrar nada, cambia `DESPLEGAR_V2` a
`false` (las pruebas siguen corriendo).

Para volver a la versión anterior si algo sale mal: consola → Cloud Run →
`kaxten-api-v2` → **Revisiones** → elige la anterior → **Administrar tráfico** →
100 %.

## Cambiar las llaves de los clientes

```bash
nano llaves.json && gcloud secrets versions add v2-llaves --data-file=llaves.json && rm llaves.json
gcloud run services update kaxten-api-v2 --region $REGION --set-secrets V2_LLAVES=v2-llaves:latest
```

## En la app (`kaxten_base`)

`DOMINIO_GLOBAL` = la URL del servicio con `/` al final, `SistemaApiFacial.LLAVE_V2` =
la llave del cliente `kaxten_base`, y `AppConfig.usarServidorFacial = true`.
