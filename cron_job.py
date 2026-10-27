"""Tareas programadas para Render o Railway: revisan la API y piden reentrenar.

No leen el volumen del modelo (en ambas plataformas un disco pertenece a un solo
servicio): todo pasa por HTTP. Solo usan la biblioteca estándar, así que también
funcionan fuera del entorno de uv.

    python cron_job.py health   # cada 15 min: /health + una predicción de prueba
    python cron_job.py retrain  # días 1 y 15: reentrena si hay datos nuevos

Variables de entorno:
    API_URL       URL pública de la API, p. ej. https://mi-api.up.railway.app
    ADMIN_TOKEN   solo para `retrain`; el mismo valor que tiene la API
    RETRAIN_POLL_SECONDS / RETRAIN_WAIT_SECONDS  (opcionales) cada cuánto
                  consultar el progreso y cuánto esperar como máximo

Código de salida: 0 = todo bien (incluye "faltan datos"), 1 = problema. La
plataforma marca la ejecución como fallida cuando es distinto de 0, y eso es lo
que se ve en su panel; no se envían notificaciones externas.
"""

import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request

# Los mensajes van a la salida estándar, que la plataforma guarda como logs.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cron")

# Un servicio gratuito de Render dormido tarda alrededor de un minuto en despertar.
REQUEST_TIMEOUT = 120
POLL_SECONDS = int(os.getenv("RETRAIN_POLL_SECONDS", "15"))
WAIT_SECONDS = int(os.getenv("RETRAIN_WAIT_SECONDS", "1800"))

# Vivienda conocida (la primera fila del CSV original) para la predicción de
# prueba: si el modelo responde algo absurdo con ella, algo está roto.
SAMPLE_HOUSE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def call(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    """Llama a la API y devuelve (código HTTP, JSON); los errores HTTP no lanzan excepción.

    urllib lanza HTTPError ante un 4xx/5xx; aquí se convierte en un resultado
    normal para que cada tarea decida qué significa cada código. Los errores de
    red (API caída, DNS) sí se propagan y los maneja main().
    """
    url = os.environ["API_URL"].rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            # Un proxy de la plataforma puede responder HTML en lugar de JSON.
            return error.code, {"detail": raw.decode(errors="replace")[:300]}


def check_health() -> int:
    """La API responde y el modelo produce una predicción razonable.

    /health solo dice que hay un modelo cargado; la predicción de prueba
    confirma que de verdad predice (que el pipeline no está corrupto, por ejemplo).
    """
    code, body = call("GET", "/health")
    if code != 200:
        logger.error("❌ /health respondió %s: %s", code, body)
        return 1

    # El encabezado evita que esta prueba cuente en el registro de predicciones.
    code, body = call("POST", "/predict", body=SAMPLE_HOUSE, headers={"X-Health-Check": "true"})
    value = body.get("predicted_median_house_value") if code == 200 else None
    # Rango amplio a propósito: solo detecta respuestas imposibles (NaN,
    # negativas, absurdamente grandes), no si el modelo es bueno.
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 10_000_000:
        logger.error("❌ Predicción de prueba inválida (HTTP %s): %s", code, body)
        return 1

    logger.info("✅ API sana · modelo %s · predicción de prueba $%s", body.get("model_version"), f"{value:,.0f}")
    return 0


def request_retrain() -> int:
    """Pide el reentrenamiento programado y espera su resultado para dejarlo en los logs.

    La API entrena en segundo plano; este proceso consulta /retrain/status
    cada POLL_SECONDS hasta que termina, así el resultado (nuevo campeón o
    no) queda escrito en los logs de esta ejecución del cron.
    """
    headers = {"X-Admin-Token": os.environ["ADMIN_TOKEN"]}
    code, body = call("POST", "/retrain?trigger=cron", headers=headers)

    # Sin datos nuevos no es un error: la API ya dejó el aviso en la interfaz.
    if code == 200 and body.get("status") == "awaiting_data":
        logger.warning("⚠️ SE NECESITAN DATOS NUEVOS: %s", body.get("message"))
        return 0
    if code == 409:
        logger.warning("Ya hay un entrenamiento en curso; no se lanza otro.")
        return 0
    # 401 (token incorrecto), 503 (modelo no listo o sin ADMIN_TOKEN), etc.
    if code != 202:
        logger.error("❌ /retrain respondió %s: %s", code, body)
        return 1

    logger.info("Reentrenamiento iniciado; esperando el resultado…")
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        code, body = call("GET", "/retrain/status")
        if code == 200 and body.get("status") != "running":
            break
    else:
        # El while terminó sin `break`: se agotó el tiempo de espera.
        logger.error("❌ El reentrenamiento no terminó en %d s", WAIT_SECONDS)
        return 1

    if body.get("status") == "failed":
        logger.error("❌ El reentrenamiento falló: %s", body.get("error"))
        return 1

    if body.get("status") == "promoted":
        decision = "nuevo campeón " + body.get("version", "")
    else:
        decision = "se mantiene el campeón"
    logger.info(
        "✅ Reentrenamiento terminado: %s · RMSE campeón %s · RMSE retador %s",
        decision,
        body.get("champion_metrics", {}).get("rmse"),
        body.get("challenger_metrics", {}).get("rmse"),
    )
    return 0


# Nombre de la tarea en la línea de comandos → función que la ejecuta.
TASKS = {"health": check_health, "retrain": request_retrain}


def main(argv: list[str]) -> int:
    """Ejecuta la tarea pedida y devuelve el código de salida del proceso."""
    if len(argv) != 2 or argv[1] not in TASKS:
        print(f"Uso: python {argv[0]} {{{'|'.join(TASKS)}}}", file=sys.stderr)
        return 2
    try:
        return TASKS[argv[1]]()
    except KeyError as error:
        # os.environ["API_URL"] o ["ADMIN_TOKEN"] sin definir.
        logger.error("❌ Falta la variable de entorno %s", error)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        # API caída, URL mal escrita, sin red o respuesta que no es JSON.
        logger.error("❌ No se pudo contactar la API: %s", error)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
