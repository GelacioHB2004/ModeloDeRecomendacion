"""
Microservicio de recomendación de productos — Dulcería Angelitos
Carga las reglas de asociación (Apriori) generadas en el notebook
01_recomendacion_productos.ipynb y las expone vía API REST para
que el carrito de la tienda pida sugerencias de venta cruzada.
"""

import os
import pickle
import logging
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recomendador")

app = Flask(__name__)

# En producción, si quieres restringir a tu dominio del frontend en vez de "*",
# usa: CORS(app, origins=["https://tu-tienda.com"])
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_recomendador.pkl")

# ---------------------------------------------------------------------------
# Carga del modelo al iniciar el servicio
# ---------------------------------------------------------------------------

modelo = None
reglas_por_antecedente = defaultdict(list)  # nombre_producto -> [ {consecuente, id, confidence, lift, support}, ... ]
mapa_nombre_a_id = {}
mapa_id_a_nombre = {}
metadata = {}


def cargar_modelo():
    """Carga el .pkl y precalcula un índice por antecedente para respuestas rápidas."""
    global modelo, mapa_nombre_a_id, mapa_id_a_nombre, metadata, reglas_por_antecedente

    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    mapa_nombre_a_id = dict(modelo["mapa_nombre_a_id"])
    mapa_id_a_nombre = dict(modelo["mapa_id_a_nombre"])
    metadata = dict(modelo["metadata"])

    reglas_por_antecedente = defaultdict(list)
    reglas_df = modelo["reglas"]

    # Convertimos el DataFrame a estructuras nativas de Python una sola vez,
    # aquí al inicio, para que el resto del servicio no dependa de pandas
    # en cada request (más rápido y evita problemas de versiones en runtime).
    for _, fila in reglas_df.iterrows():
        antecedente = str(fila["antecedente"])
        consecuente = str(fila["consecuente"])
        reglas_por_antecedente[antecedente].append({
            "producto": consecuente,
            "id_producto": mapa_nombre_a_id.get(consecuente),
            "support": float(fila["support"]),
            "confidence": float(fila["confidence"]),
            "lift": float(fila["lift"]),
        })

    # Cada lista de recomendaciones ordenada de mejor a peor (mayor lift primero)
    for antecedente in reglas_por_antecedente:
        reglas_por_antecedente[antecedente].sort(key=lambda r: r["lift"], reverse=True)

    logger.info(
        "Modelo cargado: %s reglas, %s productos con al menos una regla, entrenado con %s canastas.",
        metadata.get("total_reglas"),
        len(reglas_por_antecedente),
        metadata.get("total_canastas_entrenamiento"),
    )


cargar_modelo()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "servicio": "Recomendador de productos - Dulcería Angelitos",
        "estado": "activo",
        "endpoints": {
            "/salud": "GET - healthcheck",
            "/recomendar": "POST - recibe {\"carrito\": [id_producto, ...]} y devuelve recomendaciones",
        },
    })


@app.route("/salud", methods=["GET"])
def salud():
    return jsonify({"estado": "ok", "reglas_cargadas": metadata.get("total_reglas", 0)})


@app.route("/recomendar", methods=["POST"])
def recomendar():
    """
    Body esperado (JSON):
        { "carrito": [12, 5, 28] }     -> ids de producto ya en el carrito
    También se acepta un solo producto:
        { "producto_id": 12 }

    Respuesta:
        {
          "recomendaciones": [
            {"id_producto": 3, "nombre": "Cacahuate Grande", "confidence": 0.47, "lift": 1.26},
            ...
          ],
          "con_reglas": true
        }

    Nota importante: este microservicio solo conoce las reglas de asociación,
    no el estado actual del catálogo (Activo/Inactivo, stock, etc.). Como se
    documentó en el notebook (sección 3.5), el backend/carrito que consuma
    esta respuesta debe filtrar cualquier id_producto que ya no esté activo
    o sin stock antes de mostrarlo al cliente.
    """
    body = request.get_json(silent=True) or {}

    ids_carrito = body.get("carrito")
    if ids_carrito is None:
        producto_id = body.get("producto_id")
        ids_carrito = [producto_id] if producto_id is not None else []

    if not isinstance(ids_carrito, list) or len(ids_carrito) == 0:
        return jsonify({
            "error": "Envía 'carrito' (lista de id_producto) o 'producto_id' (un solo id) en el body JSON."
        }), 400

    try:
        ids_carrito = [int(i) for i in ids_carrito]
    except (TypeError, ValueError):
        return jsonify({"error": "Los ids de producto deben ser enteros."}), 400

    nombres_carrito = [mapa_id_a_nombre[i] for i in ids_carrito if i in mapa_id_a_nombre]

    if not nombres_carrito:
        return jsonify({
            "recomendaciones": [],
            "con_reglas": False,
            "mensaje": "Ninguno de los ids enviados existe en el modelo (¿producto nuevo sin historial?).",
        })

    # Unimos las recomendaciones de cada producto del carrito, evitando duplicados
    # y evitando recomendar algo que ya está en el carrito.
    vistos = set()
    recomendaciones = []
    for nombre in nombres_carrito:
        for regla in reglas_por_antecedente.get(nombre, []):
            nombre_reco = regla["producto"]
            id_reco = regla["id_producto"]

            if id_reco in ids_carrito:
                continue
            if nombre_reco in vistos:
                continue

            vistos.add(nombre_reco)
            recomendaciones.append({
                "id_producto": id_reco,
                "nombre": nombre_reco,
                "confidence": round(regla["confidence"], 4),
                "lift": round(regla["lift"], 4),
            })

    recomendaciones.sort(key=lambda r: r["lift"], reverse=True)

    return jsonify({
        "recomendaciones": recomendaciones,
        "con_reglas": len(recomendaciones) > 0,
    })


if __name__ == "__main__":
    # Uso local. En Render, gunicorn ejecuta la app (ver Procfile / start command).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
