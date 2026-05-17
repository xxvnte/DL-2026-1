# Tarea 2 — Deep Learning

Comparación de 4 modelos sobre imágenes de peces: **CNNArch** y **MobileNetArch** (desde cero), **ResNetModel** y **GoogLeNetModel** (transfer learning).

## Estructura

```
T2/
├── script.py            # código principal
├── requirements.txt     # dependencias
├── fish_image/          # dataset (aquí van carpetas con imagenes en .png)
├── .gitignore
└── README.md
```

## Ejecución

Desde `T2/`:

```bash
pip install -r requirements.txt
python script.py
```

Requisitos: Python 3.10+, GPU opcional (CUDA si está disponible).

## Modelos entrenados

| Modelo         | Tipo                              | Épocas |
| -------------- | --------------------------------- | ------ |
| CNNArch        | CNN propia                        | 40     |
| MobileNetArch  | MobileNetV1 (depthwise separable) | 40     |
| ResNetModel    | ResNet50 ImageNet + fine-tuning   | 25     |
| GoogLeNetModel | GoogLeNet ImageNet + fine-tuning  | 25     |

Por cada modelo: mejor checkpoint `{Modelo}_best.pth`, reporte en test, `cm_{Modelo}.png`, `per_class_{Modelo}.png`.

Gráficos globales: `convergence_scratch.png`, `convergence_transfer.png`, `convergence.png`, `training_times.png`, `final_comparison.png` + resumen en consola.
