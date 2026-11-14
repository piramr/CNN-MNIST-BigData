
import base64
import io
from pathlib import Path

import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template_string
from PIL import Image, ImageOps, UnidentifiedImageError

# ==========================================
# CONFIGURACION
# ==========================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO = BASE_DIR / "modelos" / "cnn_dropout_mnist.keras"

modelo = tf.keras.models.load_model(
    RUTA_MODELO,
    compile=False
)

print("Modelo CNN cargado correctamente.")

# ==========================================
# INTERFAZ HTML Y CSS
# ==========================================

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MNIST AI | Big Data</title>

    <style>
        :root {
            --bg: #0b1221;
            --card: #15213a;
            --card2: #202f4b;
            --border: #334362;
            --text: #f1f5ff;
            --muted: #a4b3cd;
            --accent: #6388ff;
            --green: #37d5a5;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            background:
                radial-gradient(ellipse at 50% 0%,
                    #1e3260 0%, transparent 55%),
                var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        .container {
            max-width: 920px;
            margin: 0 auto;
            padding: 24px 20px 45px;
        }

        nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 30px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 21px;
            font-weight: 800;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: #283e70;
            font-size: 24px;
        }

        .nav-tag {
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--muted);
        }

        .hero {
            text-align: center;
            padding: 40px 22px;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: linear-gradient(
                135deg, #172748, #121c32
            );
            margin-bottom: 24px;
        }

        .eyebrow {
            color: #a5b9ff;
            font-size: 12px;
            letter-spacing: 2px;
            font-weight: 800;
        }

        h1 {
            font-size: clamp(28px, 5vw, 44px);
            line-height: 1.16;
            margin: 16px auto;
            max-width: 700px;
        }

        .gradient {
            background: linear-gradient(
                90deg, #8ca7ff, #b79aff
            );
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .description {
            color: var(--muted);
            max-width: 620px;
            margin: 0 auto 24px;
            line-height: 1.7;
        }

        .badges {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .badge {
            padding: 8px 13px;
            border-radius: 30px;
            background: #26375b;
            color: #d1dcff;
            font-size: 12px;
            font-weight: 600;
        }

        .badge.green {
            background: #173e3b;
            color: #7ce7bf;
        }

        .panel {
            border: 1px solid var(--border);
            background: var(--card);
            padding: 28px;
            border-radius: 20px;
            margin-bottom: 24px;
        }

        .panel h2 {
            margin: 0 0 8px;
            font-size: 22px;
        }

        .subtitle {
            font-size: 14px;
            color: var(--muted);
            line-height: 1.6;
            margin-bottom: 23px;
        }

        .upload-zone {
            border: 2px dashed #53678f;
            border-radius: 16px;
            padding: 35px 20px;
            text-align: center;
            background: #101b31;
            transition: border-color 0.2s;
        }

        .upload-zone:hover {
            border-color: var(--accent);
        }

        .upload-icon {
            font-size: 40px;
            margin-bottom: 12px;
        }

        .upload-zone h3 {
            margin: 0 0 8px;
        }

        .upload-zone p {
            color: var(--muted);
            font-size: 13px;
        }

        input[type="file"] {
            display: block;
            width: 100%;
            max-width: 350px;
            margin: 20px auto 0;
            color: var(--muted);
        }

        input::file-selector-button {
            padding: 10px 15px;
            background: #263f74;
            border: 1px solid #4968ac;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            margin-right: 12px;
        }

        .predict-btn {
            display: block;
            width: 100%;
            margin-top: 18px;
            padding: 16px;
            background: linear-gradient(
                90deg, #4368e8, #7358d8
            );
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
        }

        .result {
            background: var(--card);
            border: 1px solid #4064a2;
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 24px;
        }

        .result h2 {
            margin-top: 0;
            text-align: center;
        }

        .images {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 24px 0;
        }

        .image-box {
            padding: 20px;
            border-radius: 14px;
            background: #101b31;
            text-align: center;
        }

        .image-box p {
            font-size: 13px;
            color: var(--muted);
            margin-top: 0;
        }

        .image-box img {
            width: 140px;
            height: 140px;
            object-fit: contain;
            border-radius: 8px;
            background: #000;
        }

        .prediction {
            text-align: center;
            background: #102b35;
            border: 1px solid #286c6c;
            border-radius: 16px;
            padding: 24px;
        }

        .prediction .number {
            font-size: 90px;
            font-weight: 800;
            color: var(--green);
            line-height: 1.2;
        }

        .confidence {
            color: #a9e9d9;
            margin: 10px 0;
        }

        .progress {
            height: 10px;
            background: #27454b;
            border-radius: 20px;
            overflow: hidden;
            max-width: 400px;
            margin: 18px auto 0;
        }

        .progress-fill {
            height: 100%;
            border-radius: 20px;
            background: linear-gradient(
                90deg, #24a98b, #60efb1
            );
        }

        .team {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 20px;
        }

        .member {
            padding: 22px 10px;
            background: var(--card2);
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border);
        }

        .avatar {
            width: 50px;
            height: 50px;
            display: grid;
            place-items: center;
            background: #354b78;
            border-radius: 50%;
            margin: 0 auto 12px;
            font-size: 22px;
        }

        .member strong {
            font-size: 14px;
        }

        .error {
            background: #51242c;
            color: #ffb5bd;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }

        footer {
            text-align: center;
            color: #8293b2;
            font-size: 12px;
            margin-top: 30px;
            line-height: 1.8;
        }

        @media (max-width: 600px) {
            .container { padding: 12px; }
            .hero { padding: 28px 15px; }
            .panel, .result { padding: 18px; }
            .images { gap: 10px; }
            .image-box { padding: 12px 5px; }
            .image-box img {
                width: 110px;
                height: 110px;
            }
            .team { grid-template-columns: 1fr; }
        }
    </style>
</head>

<body>
<div class="container">

    <nav>
        <div class="brand">
            <div class="brand-icon">🧠</div>
            MNIST AI
        </div>
        <div class="nav-tag">BIG DATA · CNN</div>
    </nav>

    <header class="hero">
        <div class="eyebrow">
            INTELIGENCIA ARTIFICIAL · DEEP LEARNING
        </div>

        <h1>
            Reconocimiento de dígitos
            con <span class="gradient">Deep Learning</span>
        </h1>

        <p class="description">
            Aplicación web que utiliza una red neuronal
            convolucional entrenada con MNIST para
            reconocer dígitos escritos a mano.
        </p>

        <div class="badges">
            <span class="badge">CNN + Dropout</span>
            <span class="badge green">
                99,12 % Accuracy en MNIST
            </span>
            <span class="badge">TensorFlow / Keras</span>
        </div>
    </header>

    <section class="panel">
        <h2>📤 Analizar imagen</h2>

        <p class="subtitle">
            Sube una imagen que contenga un único dígito
            del 0 al 9. Se aceptan archivos PNG y JPG.
        </p>

        <form method="POST"
              enctype="multipart/form-data">

            <div class="upload-zone">
                <div class="upload-icon">🖼️</div>
                <h3>Selecciona tu imagen</h3>
                <p>
                    Se convertirá a escala de grises
                    y se adaptará a 28 × 28 píxeles.
                </p>

                <input
                    type="file"
                    name="imagen"
                    accept=".png,.jpg,.jpeg"
                    required
                >
            </div>

            <button class="predict-btn" type="submit">
                ✨ Analizar dígito
            </button>
        </form>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </section>

    {% if prediccion is not none %}

    <section class="result">
        <h2>🔍 Resultado del análisis</h2>

        <div class="images">
            <div class="image-box">
                <p>Imagen original</p>
                <img
                    src="data:image/png;base64,{{ original }}"
                    alt="Imagen original"
                >
            </div>

            <div class="image-box">
                <p>Imagen procesada (28 × 28)</p>
                <img
                    src="data:image/png;base64,{{ procesada }}"
                    alt="Imagen procesada"
                    style="image-rendering:pixelated;"
                >
            </div>
        </div>

        <div class="prediction">
            <p>DÍGITO RECONOCIDO</p>

            <div class="number">
                {{ prediccion }}
            </div>

            <div class="confidence">
                Confianza: <strong>{{ confianza }} %</strong>
            </div>

            <div class="progress">
                <div class="progress-fill"
                     style="width:{{ confianza }}%">
                </div>
            </div>

            <p style="font-size:12px;color:#9bbdbb;">
                La confianza es una estimación del modelo,
                no una garantía de acierto.
            </p>
        </div>
    </section>

    {% endif %}

    <section class="panel">
        <h2>👥 Integrantes del proyecto</h2>

        <p class="subtitle">
            Desarrollo académico de una aplicación
            de reconocimiento de dígitos mediante CNN.
        </p>

        <div class="team">
            <div class="member">
                <div class="avatar">👨‍💻</div>
                <strong>Pusda Stiven</strong>
            </div>

            <div class="member">
                <div class="avatar">👨‍💻</div>
                <strong>Ramírez Pablo</strong>
            </div>

            <div class="member">
                <div class="avatar">👨‍💻</div>
                <strong>Moran Steven</strong>
            </div>
        </div>
    </section>

    <footer>
        Proyecto académico · Big Data · Deep Learning
        <p>
            Clasificación de dígitos MNIST
            mediante CNN con Dropout
        </p>
    </footer>

</div>
</body>
</html>
"""

# ==========================================
# CONVERTIR IMAGEN A BASE64
# ==========================================

def imagen_a_base64(imagen):
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ==========================================
# PREPROCESAR IMAGEN
# ==========================================

def preparar_imagen(imagen):

    imagen = ImageOps.exif_transpose(imagen)
    imagen = imagen.convert("L")

    pixeles = np.asarray(imagen, dtype=np.uint8)

    borde = np.concatenate([
        pixeles[0, :],
        pixeles[-1, :],
        pixeles[:, 0],
        pixeles[:, -1]
    ])

    if np.median(borde) > 127:
        imagen = ImageOps.invert(imagen)

    if imagen.size != (28, 28):

        valores = np.asarray(imagen)
        mascara = valores > 35

        if not np.any(mascara):
            raise ValueError(
                "No se detecto un digito visible."
            )

        filas, columnas = np.where(mascara)

        recorte = imagen.crop((
            int(columnas.min()),
            int(filas.min()),
            int(columnas.max()) + 1,
            int(filas.max()) + 1
        ))

        ancho, alto = recorte.size
        escala = min(20 / ancho, 20 / alto)

        nuevo_ancho = max(1, round(ancho * escala))
        nuevo_alto = max(1, round(alto * escala))

        recorte = recorte.resize(
            (nuevo_ancho, nuevo_alto),
            Image.Resampling.LANCZOS
        )

        lienzo = Image.new("L", (28, 28), 0)

        x = (28 - nuevo_ancho) // 2
        y = (28 - nuevo_alto) // 2

        lienzo.paste(recorte, (x, y))
        imagen = lienzo

    arreglo = np.asarray(
        imagen,
        dtype=np.float32
    ) / 255.0

    arreglo = arreglo.reshape(1, 28, 28, 1)

    return arreglo, imagen


# ==========================================
# RUTA PRINCIPAL FLASK
# ==========================================

@app.route("/", methods=["GET", "POST"])
def inicio():

    prediccion = None
    confianza = None
    original = None
    procesada = None
    error = None

    if request.method == "POST":

        archivo = request.files.get("imagen")

        if archivo is None or archivo.filename == "":
            error = "Selecciona una imagen."

        elif not archivo.filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            error = "Solo se aceptan imagenes PNG o JPG."

        else:
            try:
                imagen = Image.open(archivo.stream)
                imagen.load()

                if imagen.format not in ("PNG", "JPEG"):
                    raise ValueError(
                        "El contenido no es PNG o JPEG."
                    )

                original = imagen_a_base64(imagen)

                datos, imagen_procesada = preparar_imagen(
                    imagen
                )

                procesada = imagen_a_base64(
                    imagen_procesada
                )

                probabilidades = modelo(
                    datos,
                    training=False
                ).numpy()[0]

                prediccion = int(
                    np.argmax(probabilidades)
                )

                confianza = round(
                    float(np.max(probabilidades)) * 100,
                    2
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError
            ):
                error = (
                    "No se pudo procesar la imagen. "
                    "Prueba con un digito claro."
                )

    return render_template_string(
        HTML,
        prediccion=prediccion,
        confianza=confianza,
        original=original,
        procesada=procesada,
        error=error
    )


# ==========================================
# INICIAR SERVIDOR LOCAL
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
