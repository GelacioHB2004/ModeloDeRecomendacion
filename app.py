# app.py
#
# Microservicio Flask para servir el modelo de recomendación de productos.
# Se despliega en Render (mismo patrón que el proyecto de Bike Sharing).
#
# Endpoints:
#   GET /                                -> healthcheck
#   GET /recomendar/<id_producto>        -> recomendaciones para un producto
#   POST /recomendar-carrito             -> recomendaciones para varios productos (carrito)

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)  # permite que el frontend/backend en otro dominio lo consuma

MODELO_PATH = os.path.join(os.path.dirname(__file__), 'modelo_recomendador.pkl')

with open(MODELO_PATH, 'rb') as f:
    modelo = pickle.load(f)

print(f"Modelo cargado. Total de reglas: {modelo['metadata']['total_reglas']}")


def obtener_recomendaciones(id_producto, top_n=5):
    nombre = modelo['mapa_id_a_nombre'].get(id_producto)
    if nombre is None:
        return []

    reglas = modelo['reglas']
    coincidencias = reglas[reglas['antecedente'] == nombre].head(top_n)

    resultados = []
    for _, fila in coincidencias.iterrows():
        id_rec = modelo['mapa_nombre_a_id'].get(fila['consecuente'])
        if id_rec is None:
            continue
        resultados.append({
            'id_producto_recomendado': int(id_rec),
            'nombre': fila['consecuente'],
            'confianza': round(float(fila['confidence']), 4),
            'lift': round(float(fila['lift']), 4)
        })
    return resultados


@app.route('/', methods=['GET'])
def healthcheck():
    return jsonify({
        'status': 'ok',
        'modelo': 'recomendador_productos',
        'total_reglas': modelo['metadata']['total_reglas']
    })


@app.route('/recomendar/<int:id_producto>', methods=['GET'])
def recomendar(id_producto):
    top_n = request.args.get('top_n', default=5, type=int)
    recomendaciones = obtener_recomendaciones(id_producto, top_n)
    return jsonify({
        'id_producto_base': id_producto,
        'total_recomendaciones': len(recomendaciones),
        'recomendaciones': recomendaciones
    })


@app.route('/recomendar-carrito', methods=['POST'])
def recomendar_carrito():
    """
    Body esperado: { "ids_productos": [1, 5, 8] }
    Devuelve recomendaciones combinadas para todos los productos del carrito,
    excluyendo los que ya están en el carrito, ordenadas por lift.
    """
    data = request.get_json(silent=True) or {}
    ids_productos = data.get('ids_productos', [])
    top_n = data.get('top_n', 6)

    combinadas = {}
    for id_p in ids_productos:
        for rec in obtener_recomendaciones(id_p, top_n=10):
            id_rec = rec['id_producto_recomendado']
            if id_rec in ids_productos:
                continue
            if id_rec not in combinadas or rec['lift'] > combinadas[id_rec]['lift']:
                combinadas[id_rec] = rec

    resultado = sorted(combinadas.values(), key=lambda x: x['lift'], reverse=True)[:top_n]

    return jsonify({
        'total_recomendaciones': len(resultado),
        'recomendaciones': resultado
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
