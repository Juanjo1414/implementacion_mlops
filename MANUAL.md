# 📘 Manual de operación — API, monitoreo y reentrenamiento

Guía paso a paso para poner en marcha y operar la API de predicción de viviendas una vez que tienes el código: probarla en tu computador, desplegarla en Railway o Render, subir datos nuevos y entender qué hace el sistema cada quince días.

> Para el detalle técnico (cómo se elige el modelo, cómo se evita la fuga de datos) ver el [README](README.md#️-producción-monitoreo-y-reentrenamiento-quincenal) y los comentarios de `model_store.py`.

---

## Índice

1. [Cómo funciona en una página](#1-cómo-funciona-en-una-página)
2. [Requisitos](#2-requisitos)
3. [Probar en tu computador](#3-probar-en-tu-computador)
4. [Crear el token de administración](#4-crear-el-token-de-administración)
5. [Subir el código a GitHub](#5-subir-el-código-a-github)
6. [Desplegar en Railway](#6-desplegar-en-railway)
7. [Desplegar en Render](#7-desplegar-en-render)
8. [Verificar el despliegue](#8-verificar-el-despliegue)
9. [Operación del día a día](#9-operación-del-día-a-día)
10. [Tareas de mantenimiento](#10-tareas-de-mantenimiento)
11. [Solución de problemas](#11-solución-de-problemas)
12. [Referencia rápida](#12-referencia-rápida)

---

## 1. Cómo funciona en una página

Hay **tres servicios** que usan la misma imagen Docker:

| Servicio | Qué hace | Cuándo corre |
|----------|----------|--------------|
| **API** (`pycaret_app.py`) | Interfaz web, predicciones, recibe datos, reentrena | Siempre encendida |
| **Cron de salud** (`cron_job.py health`) | Revisa que la API responda y prediga bien | Cada 15 minutos |
| **Cron de reentrenamiento** (`cron_job.py retrain`) | Pide a la API que se reentrene | Días 1 y 15 de cada mes, 06:00 UTC (01:00 en Colombia) |

En Railway y Render son tres servicios. En Docker Compose los dos cron se juntan en un solo servicio, `cron`, que ejecuta `cron_job.py schedule`.

```
 Usuario ──► API (interfaz + modelo + volumen /data)
               ▲                ▲
               │                │
   cron salud ─┘    cron reentrenamiento ─┘
   (cada 15 min)    (días 1 y 15)
```

**El ciclo de datos:**

1. Alguien consigue el **valor real** de viviendas nuevas y lo sube como CSV en la interfaz.
2. El día 1 o 15, el cron le pide a la API que se reentrene.
   - Si **nadie subió datos**, no entrena: la interfaz muestra un aviso rojo *"Se necesitan datos nuevos"* y los logs del cron dicen `⚠️ SE NECESITAN DATOS NUEVOS`.
   - Si **hay datos**, entrena un modelo nuevo (el *retador*) y lo compara con el actual (el *campeón*) en datos que ninguno vio. Solo reemplaza al campeón si se equivoca menos.
3. Todo queda en el historial de la interfaz.

**Dónde se guarda todo:** en el volumen montado en `/data` (modelo, CSV subidos, historial y registro de predicciones). Sin volumen, cada redespliegue lo borra.

---

## 2. Requisitos

- [uv](https://docs.astral.sh/uv/) instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Una cuenta de GitHub con este repositorio.
- Una cuenta en **Railway** (plan Hobby) **o** en **Render** (plan de pago; ver [§7](#7-desplegar-en-render)).
- Opcional: Docker Desktop, para probar la imagen igual que en la nube.

---

## 3. Probar en tu computador

### 3.1 Instalar y levantar la API

```bash
uv sync
ADMIN_TOKEN=secreto uv run uvicorn pycaret_app:app --reload
```

Abre **http://localhost:8000**. La primera vez:

- Si existe `models/pycaret_housing.pkl` (lo exporta `pycaret_experimentos.ipynb`), la API lo adopta como campeón **v1**.
- Si no existe, entrena un modelo inicial con el CSV original; en unos segundos el indicador pasa de rojo a **"API en servicio"**.

El estado local se guarda en la carpeta `runtime/` (ignorada por Git). Para empezar de cero, bórrala con la API apagada.

### 3.2 Probar los cron jobs contra tu API local

En otra terminal:

```bash
export API_URL=http://localhost:8000 ADMIN_TOKEN=secreto
uv run python cron_job.py health    # ✅ API sana · modelo v1 · predicción de prueba $…
uv run python cron_job.py retrain   # ⚠️ SE NECESITAN DATOS NUEVOS (aún no subiste nada)
```

### 3.3 Probar el ciclo completo

1. Prepara un CSV de prueba (ver [§9.1](#91-preparar-el-csv-de-datos-nuevos)). Para practicar, sirve una muestra del CSV original:
   ```bash
   uv run python -c "import pandas as pd; pd.read_csv('1553768847-housing.csv').sample(200, random_state=7).to_csv('nuevos.csv', index=False)"
   ```
   > Estas filas ya están en los datos base, así que la mejora del RMSE que veas no es real; solo sirve para practicar el flujo.
2. En la interfaz escribe el token (`secreto`), elige `nuevos.csv` y pulsa **Subir datos**.
3. Ejecuta `uv run python cron_job.py retrain` y mira el resultado en la tabla **Historial de entrenamientos**.

### 3.4 Correr las pruebas

```bash
uv run pytest -v
uv run flake8 *.py tests/ --max-line-length=120
```

### 3.5 (Opcional) Todo con Docker Compose

Levanta la API, el programador de cron jobs, MLflow y Jupyter con un solo comando. Necesitas Docker Desktop abierto.

```bash
cp .env.example .env
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"   # pega el valor en ADMIN_TOKEN de .env
docker compose up -d --build
```

Usa siempre `--build`: sin él, Compose reutiliza las imágenes que ya existan aunque el código haya cambiado, y la API podría quedar desactualizada.

| Servicio | Dónde | Para qué |
|----------|-------|----------|
| `api` | http://localhost:8000 | Interfaz y API |
| `cron` | `docker compose logs -f cron` | Salud cada 15 min y reentrenamiento los días 1 y 15 (como en la nube) |
| `mlflow` | http://localhost:5050 | Experimentos y registro de reentrenamientos |
| `notebook` | http://localhost:8888 | Jupyter; el token sale en `docker compose logs notebook` |

- `.env` está en `.gitignore`: el token no se sube a GitHub. Sin `.env`, Compose se detiene con el mensaje *"Copia .env.example a .env y define ADMIN_TOKEN"*.
- Si el puerto 8888 está ocupado (por ejemplo, por un Jupyter abierto fuera de Docker), agrega `JUPYTER_PORT=8889` a `.env` y abre Jupyter en http://localhost:8889.
- Para reentrenar sin esperar al día 1 o 15: `docker compose exec cron python cron_job.py retrain`.
- Para apagar todo: `docker compose down` (los volúmenes con el modelo, los lotes y MLflow se conservan; `docker compose down -v` los borra).

Si solo quieres la imagen de la API, la misma que construyen Railway y Render:

```bash
docker build -t housing-api .
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=secreto -v housing-data:/data housing-api
```

---

## 4. Crear el token de administración

El token protege **subir datos** y **reentrenar**. Quien lo tenga puede cambiar el modelo, así que trátalo como una contraseña.

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Guárdalo en un gestor de contraseñas. Lo vas a pegar en:

- la variable `ADMIN_TOKEN` de la **API**, y
- la variable `ADMIN_TOKEN` del **cron de reentrenamiento** (el mismo valor).

El cron de salud **no** necesita token.

---

## 5. Subir el código a GitHub

Railway y Render despliegan desde GitHub. Sube la rama y fusiónala con `main` (o configura la plataforma para desplegar desde tu rama):

```bash
git push -u origin <tu-rama>
gh pr create --fill        # o crea el Pull Request desde la web
```

Cada push a la rama configurada redespliega los tres servicios automáticamente.

---

## 6. Desplegar en Railway

### 6.1 API

1. En [railway.com](https://railway.com): **New Project → Deploy from GitHub repo** → elige este repositorio.
2. Railway detecta el `Dockerfile` y lee `railway.json` (health check en `/health`, reinicio si falla).
3. En el servicio, pestaña **Variables**, agrega:
   | Variable | Valor |
   |----------|-------|
   | `ADMIN_TOKEN` | el token del [§4](#4-crear-el-token-de-administración) |
4. Clic derecho sobre el servicio (o **Settings**) → **Attach Volume** y móntalo en **`/data`**.
5. **Settings → Networking → Generate Domain**. Obtendrás algo como `https://housing-api-production.up.railway.app`.
6. Cambia el nombre del servicio a `housing-api` (**Settings → Service Name**) para que las referencias de abajo funcionen tal cual.

El primer arranque entrena el modelo inicial; el despliegue se marca como exitoso cuando `/health` responde 200.

### 6.2 Cron de salud

1. En el mismo proyecto: **New → GitHub Repo** → el mismo repositorio.
2. **Settings → Config-as-code** (archivo de configuración): `deploy/railway.health.json`.
   Si tu panel no muestra esa opción, configúralo a mano en **Settings**:
   - **Custom Start Command:** `python cron_job.py health`
   - **Cron Schedule:** `*/15 * * * *`
3. **Variables:**
   | Variable | Valor |
   |----------|-------|
   | `API_URL` | `https://${{housing-api.RAILWAY_PUBLIC_DOMAIN}}` |

   `${{housing-api.…}}` es una referencia de Railway: toma el dominio del servicio `housing-api` y se actualiza solo si cambia.
4. **No** le agregues volumen ni dominio: el proceso corre, escribe en los logs y termina.

### 6.3 Cron de reentrenamiento

Igual que el anterior, con estas diferencias:

- Config-as-code: `deploy/railway.retrain.json` (o **Start Command** `python cron_job.py retrain` y **Cron Schedule** `0 6 1,15 * *`).
- **Variables:**
  | Variable | Valor |
  |----------|-------|
  | `API_URL` | `https://${{housing-api.RAILWAY_PUBLIC_DOMAIN}}` |
  | `ADMIN_TOKEN` | `${{housing-api.ADMIN_TOKEN}}` |

> 💡 En la vista del cron, **Run now** ejecuta el job sin esperar al horario: útil para comprobar que todo está bien conectado.

---

## 7. Desplegar en Render

> ⚠️ **Costo.** El modelo (RandomForest de 138 MB) usa ~700 MB de RAM al reentrenar. Por eso `render.yaml` pide el plan **`1c-2g`** para la API; con 512 MB el reentrenamiento se queda sin memoria. Además, el disco persistente y los cron jobs solo existen en planes de pago.

1. En [dashboard.render.com](https://dashboard.render.com): **New → Blueprint** → elige este repositorio. Render lee `render.yaml` y propone tres servicios:
   - `housing-api` (web, con disco de 1 GB en `/data`)
   - `housing-health-check` (cron, cada 15 min)
   - `housing-retrain` (cron, días 1 y 15)
2. Render pide los valores marcados `sync: false`:
   | Servicio | Variable | Valor |
   |----------|----------|-------|
   | `housing-api` | `ADMIN_TOKEN` | el token del [§4](#4-crear-el-token-de-administración) |
   | `housing-retrain` | `ADMIN_TOKEN` | **el mismo** token |
   | ambos cron | `API_URL` | la URL pública de la API |

   Si aún no conoces la URL de la API, pon cualquier valor, termina el despliegue, copia la URL que aparece arriba en el servicio `housing-api` (`https://housing-api-xxxx.onrender.com`) y cámbiala en **Environment** de cada cron.
3. **Apply**. Render construye la imagen una vez por servicio.

> 💡 En cada cron, **Trigger Run** lo ejecuta en el momento.

---

## 8. Verificar el despliegue

Reemplaza `<API>` por tu URL pública.

| Comprobación | Cómo | Resultado esperado |
|--------------|------|--------------------|
| La API está viva | Abre `<API>/health` | `{"status":"healthy","model_loaded":true,"model_version":"v1"}` |
| La interfaz carga | Abre `<API>/` | Indicador verde "API en servicio" y el campeón v1 |
| El token funciona | En la interfaz, escribe el token y pulsa **Reentrenar ahora** | Mensaje "No hay datos nuevos…" (no "Token de administración inválido") |
| Cron de salud | Ejecútalo a mano (*Run now* / *Trigger Run*) y abre sus logs | `✅ API sana · modelo v1 · …` |
| Cron de reentrenamiento | Ejecútalo a mano y abre sus logs | `⚠️ SE NECESITAN DATOS NUEVOS` (normal si aún no subiste datos) |
| El volumen persiste | Redespliega la API y recarga la interfaz | El historial sigue ahí |

---

## 9. Operación del día a día

### 9.1 Preparar el CSV de datos nuevos

El archivo debe tener **encabezado** y estas columnas (el orden no importa; columnas extra se ignoran):

| Columna | Tipo | Regla |
|---------|------|-------|
| `longitude` | número | obligatorio |
| `latitude` | número | obligatorio |
| `housing_median_age` | número | ≥ 0 |
| `total_rooms` | número | ≥ 0 |
| `total_bedrooms` | número | ≥ 0, **puede ir vacío** |
| `population` | número | ≥ 0 |
| `households` | número | ≥ 0 |
| `median_income` | número | ≥ 0 (en decenas de miles de USD: `8.3` = 83 000 USD) |
| `ocean_proximity` | texto | uno de `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` |
| `median_house_value` | número | **> 0**: el valor real de la vivienda en USD |

Ejemplo:

```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity,median_house_value
-122.23,37.88,41,880,129,322,126,8.3252,NEAR BAY,452600
-118.25,34.05,30,2000,,1200,450,3.1,<1H OCEAN,310000
```

Otras reglas:

- Mínimo **10 filas** por archivo y máximo **20 MB**.
- Separador coma y punto decimal (`8.3252`, no `8,3252`). Si exportas desde Excel en español, elige *"CSV (delimitado por comas)"* y revisa el separador decimal.
- Si una fila tiene un error, se **rechaza el archivo completo** y la interfaz lista cada problema con su número de línea (la línea 1 es el encabezado). Corrige y vuelve a subirlo.

### 9.2 Subir los datos

**Desde la interfaz:** sección *Datos nuevos y reentrenamiento* → escribe el token → elige el archivo → **Subir datos**. El lote aparece en *Lotes subidos* como **Pendiente** y el aviso superior cambia a verde.

**Desde la terminal:**

```bash
curl -X POST <API>/data/upload \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: text/csv" \
  --data-binary @nuevos.csv
```

Puedes subir varios archivos antes del próximo reentrenamiento; se usan todos.

### 9.3 Qué pasa los días 1 y 15

El cron llama a `/retrain`:

| Situación | Qué hace | Qué ves |
|-----------|----------|---------|
| No hay lotes pendientes | No entrena | Aviso rojo en la interfaz; `awaiting_data` en el historial |
| Hay lotes pendientes y el retador es mejor | Lo pone a servir (v2, v3…) sin reiniciar | **Nuevo campeón** en el historial |
| Hay lotes pendientes y el retador no es mejor | Mantiene el modelo actual | **Se mantuvo el actual** |
| Error al entrenar | Mantiene el modelo actual; los lotes siguen pendientes | **Falló** con el mensaje de error |

En los dos casos con datos, los lotes quedan marcados como **Usado**: reentrenar otra vez con los mismos datos daría el mismo resultado.

### 9.4 Reentrenar sin esperar al cron

Interfaz → **Reentrenar ahora**, o:

```bash
curl -X POST "<API>/retrain" -H "X-Admin-Token: $ADMIN_TOKEN"
curl "<API>/retrain/status"     # running → promoted / kept / failed
```

### 9.5 Leer el monitoreo

En el panel *Monitoreo de predicciones*:

- **Predicciones / Últimas 24 h:** volumen de uso. Las pruebas del cron de salud no cuentan.
- **Promedio:** si cambia mucho de una semana a otra sin razón, revisa qué datos está recibiendo la API.
- **Atípicas 24 h:** porcentaje de predicciones con alguna variable a más de 3 desviaciones de los datos de entrenamiento. Si sube de forma sostenida, la realidad se está alejando de los datos del modelo: es la señal para conseguir y subir datos nuevos.

El detalle de cada predicción está en `/data/predictions.jsonl` dentro del volumen.

### 9.6 Revisar los logs de los cron

| Mensaje | Significado | Acción |
|---------|-------------|--------|
| `✅ API sana …` | Todo bien | Ninguna |
| `❌ /health respondió 503` | La API no tiene modelo cargado | Ver los logs de la API ([§11](#11-solución-de-problemas)) |
| `❌ No se pudo contactar la API` | API caída o `API_URL` mal escrita | Revisar el servicio y la variable |
| `⚠️ SE NECESITAN DATOS NUEVOS` | No había lotes para reentrenar | Subir un CSV ([§9.2](#92-subir-los-datos)) |
| `✅ Reentrenamiento terminado: nuevo campeón v3 …` | Se cambió el modelo | Ninguna |
| `❌ /retrain respondió 401` | El token del cron no coincide con el de la API | Igualar `ADMIN_TOKEN` |

Railway y Render marcan en rojo las ejecuciones que terminan con `❌`. No hay notificaciones por correo ni chat: revisa el panel de la plataforma o la interfaz.

---

## 10. Tareas de mantenimiento

### 10.1 Cambiar el horario de reentrenamiento

- **Railway:** edita `cronSchedule` en `deploy/railway.retrain.json` (o el campo *Cron Schedule* del servicio).
- **Render:** edita `schedule` del servicio `housing-retrain` en `render.yaml`.
- **Docker Compose:** agrega `RETRAIN_DAYS`, `RETRAIN_HOUR_UTC` o `HEALTH_EVERY_MINUTES` al `environment` del servicio `cron` en `compose.yml`.

Formato cron en **UTC**: `minuto hora día-del-mes mes día-de-la-semana`. Ejemplos:

| Expresión | Significado |
|-----------|-------------|
| `0 6 1,15 * *` | Días 1 y 15 a las 06:00 UTC (actual) |
| `0 6 * * 1` | Todos los lunes |
| `0 6 1 * *` | Una vez al mes |

Cron no puede expresar "cada 14 días exactos"; los días 1 y 15 son la aproximación habitual. Recuerda actualizar también el texto de la interfaz (`static/index.html`) si cambias la frecuencia.

### 10.2 Cambiar el token

1. Genera uno nuevo ([§4](#4-crear-el-token-de-administración)).
2. Cámbialo en la API y en el cron de reentrenamiento (en Railway basta con la API si el cron usa `${{housing-api.ADMIN_TOKEN}}`).
3. La plataforma redespliega; quien use la interfaz debe escribir el token nuevo.

### 10.3 Usar un modelo nuevo del cuaderno

**En local / Docker Compose:** ejecuta `pycaret_experimentos.ipynb` (exporta `models/pycaret_housing.pkl`) y reinicia la API (`docker compose restart api`). Al arrancar detecta que el archivo es nuevo y lo adopta como siguiente versión.

**En la nube:** `models/` no está en Git, así que el `.pkl` no llega a la imagen. El modelo inicial en la nube se crea con `default_pipeline()` de `model_store.py` (la misma estructura que exporta el cuaderno) y los reentrenamientos clonan al campeón vigente. Si en el cuaderno eliges otro algoritmo y quieres usarlo en la nube:

1. Cambia `default_pipeline()` en `model_store.py` para que construya ese pipeline y despliega.
2. Borra el campeón del volumen para que se entrene de nuevo: abre una terminal en el servicio (Railway: `railway ssh`; Render: pestaña **Shell**) y ejecuta `rm /data/models/champion.pkl`. Luego reinicia el servicio.

El historial y los lotes subidos se conservan; el nuevo modelo se entrena con la partición base más los lotes ya usados.

### 10.4 Empezar de cero

Borra el contenido del volumen (`rm -rf /data/*` desde la terminal del servicio) y reinicia la API. Se pierden los lotes, el historial y el registro de predicciones.

---

## 11. Solución de problemas

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| `/health` responde 503 varios minutos después de desplegar | El modelo inicial sigue entrenando o falló | Logs de la API: busca `Campeón v1 listo` o `No se pudo preparar el modelo` |
| La API se reinicia sola al reentrenar (Render) | Sin memoria (plan de 512 MB) | Plan `1c-2g` o superior |
| Tras redesplegar desaparecen historial y lotes | No hay volumen o no está en `/data` | Montar el volumen en `/data` ([§6.1](#61-api) / `render.yaml`) |
| "Configure ADMIN_TOKEN para habilitar la administración" | La API no tiene la variable | Agregar `ADMIN_TOKEN` a la API |
| "Token de administración inválido" | Token distinto o con espacios al copiarlo | Volver a pegarlo sin espacios |
| El CSV se rechaza por "valores no numéricos" | Comas decimales o separador `;` | Exportar con punto decimal y coma como separador |
| Siempre sale "Se mantuvo el actual" | Los datos nuevos no mejoran el error en el holdout | Normal con pocos datos; acumula más lotes |
| El cron de salud tarda ~1 min la primera vez (Render) | El servicio estaba dormido | Normal; el cron espera hasta 120 s |
| `/data/upload` responde 404 en Docker Compose | La imagen de la API es anterior a estos cambios | `docker compose up -d --build` |
| `port is already allocated` / `address already in use` en 8888 | Otro Jupyter usa ese puerto | `JUPYTER_PORT=8889` en `.env` |
| Advertencia `X does not have valid feature names` en los logs | Viene del pipeline exportado por PyCaret | Inofensiva |

---

## 12. Referencia rápida

### Rutas de la API

| Ruta | Token | Para qué |
|------|:-----:|----------|
| `GET /` | | Interfaz web |
| `GET /docs` | | Swagger: probar la API desde el navegador |
| `GET /health` | | Salud: 200 con versión, 503 sin modelo |
| `POST /predict` | | Predicción de una vivienda (JSON) |
| `GET /status` | | Campeón, lotes, historial, avisos |
| `GET /monitor/summary` | | Estadísticas de predicciones |
| `POST /data/upload` | ✅ | Subir CSV (`Content-Type: text/csv`) |
| `POST /retrain` | ✅ | Reentrenar (202 iniciado, 200 sin datos, 409 en curso) |
| `GET /retrain/status` | | Progreso o resultado del último |

### Variables de entorno

| Variable | Servicio | Valor por defecto | Descripción |
|----------|----------|-------------------|-------------|
| `ADMIN_TOKEN` | API, cron retrain | *(ninguno)* | Token de administración; sin él la administración queda deshabilitada. En Docker Compose se lee de `.env` |
| `DATA_DIR` | API | `/data` en Docker, `runtime` en local | Dónde se guarda el estado |
| `SEED_MODEL_PATH` | API | `models/pycaret_housing.pkl` | `.pkl` del cuaderno que se adopta si es nuevo |
| `BASE_DATA_PATH` | API | `1553768847-housing.csv` | Datos base de entrenamiento y evaluación |
| `MLFLOW_TRACKING_URI` | API | *(ninguno)* | Si se define, cada reentrenamiento se registra en MLflow |
| `PORT` | API | `8000` (Railway y Render lo asignan) | Puerto HTTP |
| `API_URL` | ambos cron | *(ninguno)* | URL pública de la API |
| `RETRAIN_POLL_SECONDS` | cron retrain | `15` | Cada cuánto consulta el progreso |
| `RETRAIN_WAIT_SECONDS` | cron retrain | `1800` | Espera máxima antes de marcar error |
| `HEALTH_EVERY_MINUTES`, `RETRAIN_DAYS`, `RETRAIN_HOUR_UTC` | cron de Compose | `15`, `1,15`, `6` | Horario de `cron_job.py schedule` |

### Archivos

| Archivo | Contenido |
|---------|-----------|
| `pycaret_app.py` | Rutas de la API |
| `model_store.py` | Campeón, lotes, reentrenamiento y monitoreo |
| `cron_job.py` | Las dos tareas programadas y el programador `schedule` de Docker Compose |
| `compose.yml`, `.env.example` | Entorno local completo con Docker Compose |
| `static/index.html` | Interfaz web |
| `Dockerfile` | Imagen (la etapa `api` es la última) |
| `railway.json`, `deploy/railway.*.json` | Configuración de Railway |
| `render.yaml` | Blueprint de Render |
| `tests/test_retraining.py` | Pruebas del ciclo completo |
