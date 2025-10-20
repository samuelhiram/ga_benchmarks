#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) benchmark runner
- Optimiza 5 funciones de prueba usando un modelo de ISLAS:
  Para CADA función se lanzan N hilos (islas) en paralelo; se toma el mejor.
- Genera gráficos de superficies (2D) y convergencia.
- Construye una presentación HTML con Portada, Índice, Introducción, Desarrollo, Resultados, Conclusiones y Referencias.

Requisitos:
  - Python 3.9+
  - numpy, matplotlib
Uso:
  python3 main.py
"""

import os, math, time, json, random, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")  # para entornos sin GUI
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

# -----------------------------
# Configuración y utilidades GA
# -----------------------------

@dataclass
class GAParams:
    seed: int = 42
    dims: int = 2
    population: int = 80
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1        # prob por gen
    mutation_sigma_ratio: float = 0.1 # % del rango
    tournament_k: int = 3
    elite_size: int = 2
    stagnation_patience: int = 60     # termina si no mejora tras N generaciones
    islands: int = 5                  # <-- N hilos por función (ISLAS)

# Semillas globales (no usadas dentro de las islas; solo por compat.)
random.seed(GAParams.seed)
np.random.seed(GAParams.seed)

# ====== Operadores con RNG LOCAL (para reproducibilidad en paralelo) ======

def tournament_select_rng(pop, fitness, k, rng):
    n = len(pop)
    idxs = rng.integers(0, n, size=k)
    best_i = idxs[np.argmin(fitness[idxs])]
    return pop[best_i]

def arithmetic_crossover_rng(p1, p2, rng):
    """ Cruce aritmético (dos hijos) usando RNG local. """
    alpha = rng.random(p1.shape)
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

def gaussian_mutation_rng(x, low, high, pm, sigma, rng):
    mask  = rng.random(x.shape) < pm
    noise = rng.normal(0.0, sigma, size=x.shape)
    y = np.where(mask, x + noise, x)
    return np.clip(y, low, high)

def initialize_population_rng(pop_size, dims, low, high, rng):
    return rng.uniform(low, high, size=(pop_size, dims))

# ====== Benchmarks ======

def sphere(x):
    return np.sum(x**2, axis=-1)

def rastrigin(x):
    return np.sum(x**2 - 10*np.cos(2*math.pi*x) + 10, axis=-1)

def rosenbrock(x):
    # \sum_{i=1}^{d-1}[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
    return np.sum(100.0*(x[...,1:] - x[...,:-1]**2)**2 + (1 - x[...,:-1])**2, axis=-1)

def ackley(x):
    d = x.shape[-1]
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=-1)/d)) \
           - np.exp(np.sum(np.cos(2*math.pi*x), axis=-1)/d) + 20 + math.e

def griewank(x):
    d = x.shape[-1]
    sum_part = np.sum(x**2, axis=-1)/4000.0
    i = np.arange(1, d+1)
    cos_part = np.prod(np.cos(x / np.sqrt(i)), axis=-1)
    return sum_part - cos_part + 1

FUNC_INFO = {
    "Sphere": {
        "f": sphere,
        "bounds": (-5.12, 5.12),
        "global_min_x": "x* = (0,...,0)",
        "global_min_f": 0.0,
        "latex": "f(\\mathbf{x}) = \\sum_{i=1}^{d} x_i^2"
    },
    "Rastrigin": {
        "f": rastrigin,
        "bounds": (-5.12, 5.12),
        "global_min_x": "x* = (0,...,0)",
        "global_min_f": 0.0,
        "latex": "f(\\mathbf{x}) = 10d + \\sum_{i=1}^d \\left(x_i^2 - 10\\cos(2\\pi x_i)\\right)"
    },
    "Rosenbrock": {
        "f": rosenbrock,
        "bounds": (-2.0, 2.0),
        "global_min_x": "x* = (1,...,1)",
        "global_min_f": 0.0,
        "latex": "f(\\mathbf{x}) = \\sum_{i=1}^{d-1} \\left[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2\\right]"
    },
    "Ackley": {
        "f": ackley,
        "bounds": (-5.0, 5.0),
        "global_min_x": "x* = (0,...,0)",
        "global_min_f": 0.0,
        "latex": "f(\\mathbf{x}) = -20\\exp\\Big(-0.2\\sqrt{\\tfrac{1}{d}\\sum x_i^2}\\Big) - \\exp\\Big(\\tfrac{1}{d}\\sum \\cos(2\\pi x_i)\\Big) + 20 + e"
    },
    "Griewank": {
        "f": griewank,
        "bounds": (-6.0, 6.0),
        "global_min_x": "x* = (0,...,0)",
        "global_min_f": 0.0,
        "latex": "f(\\mathbf{x}) = 1 + \\sum\\tfrac{x_i^2}{4000} - \\prod \\cos\\Big(\\tfrac{x_i}{\\sqrt{i}}\\Big)"
    }
}

def ensure_dirs(base):
    out_dir = os.path.join(base, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def plot_surface_2d(func_name, f, low, high, out_path):
    x = np.linspace(low, high, 150)
    y = np.linspace(low, high, 150)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    Z = f(XY)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_title(f"{func_name} (superficie 2D)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f(x)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_convergence(func_name, best_history, out_path):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(best_history)
    plt.yscale("log")
    plt.xlabel("Generación")
    plt.ylabel("Mejor f(x) (escala log)")
    plt.title(f"Convergencia - {func_name}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# =======================
#  GA: una ISLA (1 hilo)
# =======================
def run_ga_island(name, params: GAParams, seed_seq):
    """ Ejecuta un GA independiente (una 'isla') y retorna su mejor trayectoria. """
    rng = np.random.default_rng(seed_seq)
    f = FUNC_INFO[name]["f"]
    low, high = FUNC_INFO[name]["bounds"]
    dims = params.dims
    sigma = params.mutation_sigma_ratio * (high - low)

    # Inicialización
    pop = initialize_population_rng(params.population, dims, low, high, rng)
    fitness = f(pop)
    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    best_history = [best_fit]
    stagnation = 0

    for gen in range(1, params.generations+1):
        new_pop = []
        # Elitismo
        elite_idx = np.argsort(fitness)[:params.elite_size]
        elites = pop[elite_idx]

        while len(new_pop) < params.population - params.elite_size:
            p1 = tournament_select_rng(pop, fitness, params.tournament_k, rng)
            p2 = tournament_select_rng(pop, fitness, params.tournament_k, rng)

            if rng.random() < params.crossover_rate:
                c1, c2 = arithmetic_crossover_rng(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = gaussian_mutation_rng(c1, low, high, params.mutation_rate, sigma, rng)
            c2 = gaussian_mutation_rng(c2, low, high, params.mutation_rate, sigma, rng)

            new_pop.append(c1)
            if len(new_pop) < params.population - params.elite_size:
                new_pop.append(c2)

        pop = np.vstack([elites] + new_pop)
        fitness = f(pop)

        gen_best_idx = int(np.argmin(fitness))
        gen_best = pop[gen_best_idx].copy()
        gen_best_fit = float(fitness[gen_best_idx])

        if gen_best_fit + 1e-12 < best_fit:
            best_fit = gen_best_fit
            best = gen_best
            stagnation = 0
        else:
            stagnation += 1

        best_history.append(best_fit)

        if stagnation >= params.stagnation_patience:
            break

    return {
        "best_x": best.tolist(),
        "best_f": best_fit,
        "best_history": best_history,
        "gens_effective": len(best_history) - 1
    }

# ==================================================
# Ejecuta N islas EN PARALELO para una sola función
# ==================================================
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_function_with_islands(name, params: GAParams, base_out, seed_for_function):
    """ Lanza params.islands hilos (islas), cada uno con su sub-semilla; usa el mejor. """
    # Spawnear sub-semillas para islas (determinista)
    ss_master = np.random.SeedSequence(seed_for_function)
    island_seeds = ss_master.spawn(params.islands)

    # Paralelizar islas
    islands_results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=params.islands) as ex:
        futures = [ex.submit(run_ga_island, name, params, island_seeds[i])
                   for i in range(params.islands)]
        for fut in as_completed(futures):
            islands_results.append(fut.result())
    elapsed = time.time() - start

    # Elegir la mejor isla
    best_idx = int(np.argmin([r["best_f"] for r in islands_results]))
    best = islands_results[best_idx]

    # Gráficas (una por función, usando el histórico del mejor)
    f = FUNC_INFO[name]["f"]
    low, high = FUNC_INFO[name]["bounds"]
    surf_path = os.path.join(base_out, f"{name}_surface.png")
    conv_path = os.path.join(base_out, f"{name}_convergence.png")
    plot_surface_2d(name, f, low, high, surf_path)
    plot_convergence(name, best["best_history"], conv_path)

    # Armar resultado en el MISMO formato de antes
    return {
        "function": name,
        "dims": params.dims,
        "bounds": [low, high],
        "population": params.population,            # mismo campo, misma semántica
        "generations": params.generations,
        "crossover_rate": params.crossover_rate,
        "mutation_rate": params.mutation_rate,
        "mutation_sigma_ratio": params.mutation_sigma_ratio,
        "tournament_k": params.tournament_k,
        "elite_size": params.elite_size,
        "stagnation_patience": params.stagnation_patience,
        "best_x": best["best_x"],
        "best_f": float(best["best_f"]),
        "evaluations": int(params.population * (len(best["best_history"]))),
        "best_history": [float(v) for v in best["best_history"]],
        "surface_image": os.path.basename(surf_path),
        "convergence_image": os.path.basename(conv_path),
        "global_optimum": {
            "x": FUNC_INFO[name]["global_min_x"],
            "f": FUNC_INFO[name]["global_min_f"]
        },
        "latex": FUNC_INFO[name]["latex"],
        "elapsed_seconds": round(elapsed, 3),
        "islands_used": params.islands
    }

# ========= HTML (idéntico al original) =========

def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_presentation_html(out_dir, meta, results):
    rows = []
    for r in results:
        rows.append(f"""
        <tr>
          <td>{r['function']}</td>
          <td>{r['dims']}</td>
          <td>[{r['bounds'][0]}, {r['bounds'][1]}]</td>
          <td>{r['population']}</td>
          <td>{len(r['best_history'])-1}</td>
          <td>{r['best_f']:.6g}</td>
          <td>{html_escape(str(np.round(r['best_x'], 6).tolist()))}</td>
        </tr>
        """)
    table_html = "".join(rows)

    sections = []
    # Portada
    sections.append(f"""
    <section>
      <h1>{html_escape(meta['title'])}</h1>
      <h3>{html_escape(meta['subtitle'])}</h3>
      <p><strong>Autor:</strong> {html_escape(meta['author'])}</p>
      <p><strong>Fecha:</strong> {html_escape(meta['date'])}</p>
    </section>
    """)
    # Índice
    sections.append(f"""
    <section>
      <h2>Contenido</h2>
      <ol>
        <li>Introducción</li>
        <li>Desarrollo</li>
        <li>Resultados</li>
        <li>Conclusiones</li>
        <li>Referencias</li>
      </ol>
    </section>
    """)
    # Introducción (idéntica en estructura; el meta['workers'] lo mapeamos a #islas)
    sections.append(f"""
    <section>
      <h2>Introducción</h2>
      <p>Se resuelven en paralelo cinco funciones benchmark de optimización continua (Sphere, Rastrigin, Rosenbrock, Ackley y Griewank) empleando un Algoritmo Genético (AG) con codificación real y evaluación en hilos. Se reportan parámetros, imágenes de las funciones (en 2D), convergencia y los mejores valores hallados.</p>
      <p><strong>Hilos detectados en el equipo:</strong> {meta['cpu_cores']} (usando {meta['workers']} trabajadores).</p>
      <pre style="white-space:pre-wrap">{html_escape(json.dumps(meta['ga_params'], indent=2))}</pre>
    </section>
    """)
    # Desarrollo: por función
    dev_parts = []
    for r in results:
        dev_parts.append(f"""
        <article>
          <h3>{r['function']}</h3>
          <p><strong>Función (latex):</strong> {html_escape(r['latex'])}</p>
          <p><strong>Dimensiones:</strong> {r['dims']} |
             <strong>Dominio:</strong> [{r['bounds'][0]}, {r['bounds'][1]}] |
             <strong>Óptimo global:</strong> {html_escape(str(r['global_optimum']))}
          </p>
          <div style="display:flex;gap:24px;flex-wrap:wrap">
            <figure>
              <img src="./output/{r['surface_image']}" alt="surface" style="max-width:420px">
              <figcaption>Superficie (2D)</figcaption>
            </figure>
            <figure>
              <img src="./output/{r['convergence_image']}" alt="convergence" style="max-width:420px">
              <figcaption>Convergencia (mejor f(x) por generación)</figcaption>
            </figure>
          </div>
          <details>
            <summary>Parámetros usados del AG</summary>
            <pre style="white-space:pre-wrap">{html_escape(json.dumps({k:v for k,v in r.items() if k in ['population','dims','bounds','generations','crossover_rate','mutation_rate','mutation_sigma_ratio','tournament_k','elite_size','stagnation_patience']}, indent=2))}</pre>
          </details>
          <p><strong>Mejor f(x):</strong> {r['best_f']:.6g}</p>
          <p><strong>Mejor x encontrado:</strong> {html_escape(str(np.round(r['best_x'], 6).tolist()))}</p>
        </article>
        """)
    sections.append(f"""
    <section>
      <h2>Desarrollo</h2>
      {"".join(dev_parts)}
    </section>
    """)
    # Resultados (tabla)
    sections.append(f"""
    <section>
      <h2>Resultados</h2>
      <table border="1" cellpadding="6" cellspacing="0">
        <thead>
          <tr>
            <th>Función</th><th>Dim</th><th>Dominio</th><th>Población</th>
            <th>Generaciones reales</th><th>Mejor f(x)</th><th>Mejor x</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </section>
    """)
    # Conclusiones
    sections.append(f"""
    <section>
      <h2>Conclusiones</h2>
      <ul>
        <li>El AG con codificación real y cruce aritmético converge de forma estable en funciones suaves (p.ej., Sphere) y muestra comportamientos razonables en funciones altamente multimodales (Rastrigin, Ackley).</li>
        <li>El uso de hilos permite explorar varias funciones en paralelo, reduciendo el tiempo total de ejecución en máquinas con múltiples núcleos.</li>
        <li>Parámetros como la tasa de mutación y el tamaño de población impactan fuertemente la calidad de la solución y la velocidad de convergencia.</li>
        <li>El elitismo pequeño (2) ayuda a preservar buenos individuos y evita pérdidas por deriva aleatoria.</li>
      </ul>
    </section>
    """)
    # Referencias
    sections.append(f"""
    <section>
      <h2>Referencias</h2>
      <ol>
        <li>De Jong, K.A. (1975). <em>Analysis of the Behavior of a Class of Genetic Adaptive Systems</em>.</li>
        <li>Back, T. (1996). <em>Evolutionary Algorithms in Theory and Practice</em>. Oxford University Press.</li>
        <li>Benchmark functions: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank (literatura estándar de optimización y metaherísticas).</li>
        <li>Goldberg, D.E. (1989). <em>Genetic Algorithms in Search, Optimization and Machine Learning</em>. Addison-Wesley.</li>
      </ol>
    </section>
    """)

    html = f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{html_escape(meta['title'])}</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; line-height: 1.45; }}
        h1, h2, h3 {{ margin-top: 0.6em; }}
        section {{ margin-bottom: 40px; }}
        figure {{ margin: 0; }}
        img {{ border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.1); }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background: #f3f4f6; }}
        td, th {{ text-align: left; }}
        details {{ margin-top: 10px; }}
        pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow-x:auto; }}
      </style>
    </head>
    <body>
      {"".join(sections)}
    </body>
    </html>
    """
    with open(os.path.join(out_dir, "presentacion.html"), "w", encoding="utf-8") as f:
        f.write(html)

# ==========================
# MAIN: funciones secuenciales,
#       islas (hilos) por función
# ==========================
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "output")
    os.makedirs(out_dir, exist_ok=True)

    params = GAParams()  # valores por defecto
    cpu_cores = os.cpu_count() or 1

    # Semillas por función (deterministas)
    func_names = list(FUNC_INFO.keys())
    ss_functions = np.random.SeedSequence(params.seed)
    func_seeds = ss_functions.spawn(len(func_names))

    results = []
    start = time.time()

    # Ejecutamos CADA función, lanzando N islas en paralelo dentro de cada una
    for i, name in enumerate(func_names):
        res = run_function_with_islands(name, params, out_dir, seed_for_function=func_seeds[i].entropy)
        results.append(res)

    total_time = time.time() - start
    results.sort(key=lambda r: r["function"])

    # Persistir JSON y CSV (idénticos)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    import csv
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["function","dims","bounds","population","generations_effective","best_f","best_x"])
        for r in results:
            w.writerow([
                r["function"], r["dims"], f"[{r['bounds'][0]}, {r['bounds'][1]}]",
                r["population"], len(r["best_history"])-1,
                f"{r['best_f']:.10g}", np.array2string(np.array(r["best_x"]), precision=6, separator=", ")
            ])

    # Nota: para conservar la INTRO sin romper formato,
    # usamos 'workers' = #islas (hilos) por función.
    meta = {
        "title": "Optimización Paralela con Algoritmo Genético (5 Funciones Benchmark)",
        "subtitle": "Hilos por función (Modelo de Islas) + GA Real-Coded | Python",
        "author": "Equipo / Estudiante",
        "date": time.strftime("%Y-%m-%d"),
        "cpu_cores": cpu_cores,
        "workers": params.islands,              # mostrado en la intro
        "ga_params": vars(params),
        "total_seconds": round(total_time, 3)
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    build_presentation_html(out_dir, meta, results)

    # Consola final
    print("==== RESUMEN ====")
    print(f"Hilos (CPU detectados): {cpu_cores} | Hilos por función (islas): {params.islands}")
    print("GA params:", params)
    print(f"Tiempo total (s): {meta['total_seconds']}")
    for r in results:
        print(f"{r['function']}: best_f={r['best_f']:.6g} dim={r['dims']} gens={len(r['best_history'])-1}")

if __name__ == "__main__":
    main()
