# GA Benchmarks en Paralelo (Python)

Optimiza **5 funciones** (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank) en **paralelo con hilos** usando un **Algoritmo Genético** (codificación real). Genera gráficos y una **presentación HTML** con todos los elementos solicitados.

## Requisitos

- Python 3.9+
- Paquetes: `numpy`, `matplotlib`

## Ejecutar

```bash
python3 main.py
```

Esto:

- Ejecuta los 5 benchmarks **en paralelo** (hasta 5 hilos).
- Crea imágenes en `output/` (superficies 2D y convergencia).
- Guarda resultados en `output/results.json` y `output/results.csv`.
- Genera la **presentación** `presentacion.html` en la carpeta raíz del proyecto.

## Parámetros por defecto del AG

```json
{
  "seed": 42,
  "dims": 2,
  "population": 80,
  "generations": 200,
  "crossover_rate": 0.9,
  "mutation_rate": 0.1,
  "mutation_sigma_ratio": 0.1,
  "tournament_k": 3,
  "elite_size": 2,
  "stagnation_patience": 60
}
```

> _Nota:_ Puedes modificar estos valores directamente en `main.py` dentro de la clase `GAParams`.

## ¿Qué se entrega?

- Código Python listo para ejecutar.
- Presentación HTML (`presentacion.html`) con:
  - **Portada, Índice, Introducción**
  - **Desarrollo** (definición, imagen de la función, parámetros, óptimo global, etc.)
  - **Tabla de resultados**
  - **Conclusiones y Referencias**
- Carpeta `output/` con todas las figuras y los CSV/JSON.

---

### Sobre los hilos

Usamos `ThreadPoolExecutor` con `max_workers = min(5, os.cpu_count())`. Cada función benchmark se optimiza en un **hilo** independiente. Esto permite aprovechar múltiples núcleos cuando están disponibles.

### Nota técnica

Aunque Python tiene el GIL, este enfoque paraleliza **las funciones (tareas)** entre sí y deja a `numpy` ejecutar operaciones vectorizadas eficientes. Suficiente y simple para un proyecto académico de GA.
