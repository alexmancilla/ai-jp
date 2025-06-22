# Handwritten Digit Generator

Una aplicación web que genera imágenes de dígitos escritos a mano usando un Variational Autoencoder (VAE) entrenado en el dataset MNIST.

## Características

- Genera imágenes de dígitos del 0 al 9
- Interfaz web intuitiva con Streamlit
- Modelo VAE condicional para generación controlada
- Visualización en tiempo real de las imágenes generadas

## Instalación

1. Clona este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación localmente:

```bash
streamlit run digit_generation_app.py
```

## Deploy en Streamlit Cloud

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio de GitHub
4. Configura el archivo principal como `digit_generation_app.py`
5. ¡Listo! Tu app estará disponible online

## Requisitos

- Python 3.8+
- PyTorch
- Streamlit
- Matplotlib
- NumPy

## Nota

Asegúrate de tener el archivo de pesos del modelo `vae_mnist.pth` en el directorio raíz para que la aplicación funcione correctamente.
