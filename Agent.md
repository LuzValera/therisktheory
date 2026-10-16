# Arquitectura del Proyecto y Plan de Desarrollo Semestral — TheRiskTheory

> **Lema Rector:** *"AI-assisted development. Human-validated mathematics."*  
> **Repositorio:** `TheRiskTheory`  
> **Entorno de Ejecución Canónico:** Python >= 3.11 (validado en Python 3.12.15 vía `uv`)  
> **Mantenedor Principal y Autoridad de Fusión:** `@luispitta`

---

## 🚨 DIRECTIVA PRIMARIA (PRIMARY DIRECTIVE): ENGLISH LANGUAGE MANDATE

> ### 🛑 **MANDATORY POLICY: 100% ENGLISH FOR CODE & DOCUMENTATION**
> **All source code, identifiers, comments, docstrings, tests, commit messages, and repository documentation MUST be written strictly in English.**
> 
> * **Code Identifiers:** Package, module, class, function, variable, parameter, and constant names must be in English.
> * **Inline & Block Comments:** All explanatory comments within the codebase must be written strictly in English.
> * **Docstrings:** All docstrings (following the Google Python Style Guide) must be written exclusively in English.
> * **Tests & Assertions:** Test functions, fixture names, custom assertions, and failure messages must be in English.
> * **Exceptions & Log Messages:** All runtime error messages, warnings, and exception texts (`ValueError`, `TypeError`, etc.) must be in English.
> * **Git & GitHub Artifacts:** Branch names (`student/<topic>/<function>`), commit messages, PR titles, descriptions, and peer review feedback must be in English.
> 
> *Failure to comply (e.g., submitting comments, variable names, or docstrings in Spanish or any language other than English) is grounds for immediate rejection during peer review and maintainer review.*

---

## 📌 Puntos Clave del Proyecto (Key Highlights)

* **[KEY-1] Separación Matemática y de Datos:** El núcleo matemático depende únicamente de **NumPy** y **SciPy**. Ningún DataFrame (`pandas`, `Polars`, `Spark`) puede formar parte de las funciones matemáticas centrales.
* **[KEY-2] Vectorización Estricta:** Prohibición expresa de bucles recursivos o iteraciones fila a fila (`for row in ...`). Todos los cálculos deben operar sobre tensores vectorizados mediante operaciones nativas de NumPy o rutinas optimizadas de SciPy.
* **[KEY-3] Paridad de Backends:** Los adaptadores perimétricos para `pandas`, `Polars` y `PySpark` deben reproducir de forma idéntica los resultados de NumPy bajo tolerancias numéricas documentadas:
  $$\text{NumPy} \approx \text{pandas} \approx \text{Polars} \approx \text{Spark}$$
* **[KEY-4] Ciclo de Validación en Tres Estados:** Cada función matemática transita obligatoriamente por `experimental` $\rightarrow$ `reviewed` $\rightarrow$ `mathematically_validated`. Ningún desarrollador puede validar humanamente su propio código.
* **[KEY-5] Rol de la Inteligencia Artificial:** La IA es una herramienta de asistencia técnica (código, refactorización, tests, docstrings), pero **nunca** una fuente matemática ni un validador formal de fórmulas o teoremas.
* **[KEY-6] Gobernanza Centralizada:** La rama `master` está protegida. Todo Pull Request exige revisión por pares y aprobación final explícita de **`@luispitta`**.

---

## 1. Arquitectura del Sistema

### 1.1 Diagrama de Capas

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE ADAPTADORES                             │
│       therisktheory.backends (pandas, Polars, PySpark / Arrow)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ consume (sin alterar matemáticas)
┌───────────────────────────────────▼────────────────────────────────────┐
│                  NÚCLEO MATEMÁTICO CANÓNICO (CORE)                     │
│  therisktheory/                                                        │
│  ├── solvency_pricing/       ├── severity_aggregate/                  │
│  ├── finetti_ruin/           ├── credibility/                         │
│  └── frequency/              └── collective_risk/                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ opera exclusivamente sobre
┌───────────────────────────────────▼────────────────────────────────────┐
│                    MOTOR COMPUTACIONAL DE BAJO NIVEL                   │
│                       NumPy 2.x  +  SciPy 1.13+                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Estructura del Repositorio

```text
therisktheory/
├── src/
│   └── therisktheory/
│       ├── __init__.py
│       ├── py.typed                        # Marcador PEP 561 para comprobación estática
│       ├── backends/                       # Adaptadores opcionales (DataFrames / Big Data)
│       ├── solvency_pricing/               # Track 1: Solvencia, primas y retención
│       ├── finetti_ruin/                   # Track 2: Teoría de De Finetti y ruina
│       ├── frequency/                      # Track 3: Frecuencia y procesos de conteo
│       ├── severity_aggregate/             # Track 4: Severidad, convolución y simulación
│       ├── credibility/                    # Track 5: Credibilidad clásica, bayesiana y reservas
│       └── collective_risk/                # Track 6: Modelo colectivo y ec. de Lundberg
├── tests/                                  # Tests unitarios, vectorizados y de fronteras
├── docs/                                   # Documentación técnica y SYLLABUS.md
├── examples/                               # Scripts ejecutables de demostración
├── notebooks/                              # Cuadernos pedagógicos reproducibles
├── references/                             # Bibliografía y referencias analíticas
├── .venv/                                  # Entorno virtual aislado (Python 3.12)
├── pyproject.toml                          # Configuración del paquete y herramientas (Hatchling, Ruff, Pytest)
├── CONTRIBUTING.md                         # Guía de contribución
├── GOVERNANCE.md                           # Políticas de gobernanza y autorizaciones
└── Agent.md                                # Arquitectura integral y plan de desarrollo
```

---

## 2. Los 6 Frentes de Desarrollo en Paralelo (Tracks)

El programa se descompone en 6 módulos independientes para permitir el trabajo concurrente de 6 colaboradores:

### Track 1: Solvencia y Tarifación (`therisktheory.solvency_pricing`)
* **Alcance:**
  * Evaluación actuarial de la solvencia de la entidad aseguradora.
  * Principio de compensación de riesgos y prima pura: $\Pi = \mathbb{E}[N] \cdot \mathbb{E}[X]$.
  * Recargos de seguridad (esperanza, varianza, desviación estándar).
  * Capital de garantía y coeficiente de seguridad actuarial.
  * Principios de cálculo de primas y sus propiedades matemáticas formales.
  * Capacidad de retención financiera y esquemas de reaseguro proporcional / no proporcional.
  * Modelado de inflación en carteras de pólizas.

### Track 2: De Finetti y Teoría de la Ruina (`therisktheory.finetti_ruin`)
* **Alcance:**
  * Teoría de juegos justos e injustos y modelo clásico de ruina del jugador.
  * Fundamentos y objetivos de la teoría de De Finetti.
  * Índice de seguridad y fondo de reserva de equilibrio.
  * Beneficio de De Finetti, prima de De Finetti vs. prima pura.
  * Teorema formal de De Finetti y derivación analítica de probabilidades de ruina.
  * Determinación de la retención óptima en reaseguro proporcional bajo criterios de De Finetti.

### Track 3: Frecuencia de Siniestros (`therisktheory.frequency`)
* **Alcance:**
  * Modelado estocástico del conteo de reclamos $N$.
  * Tasa instantánea de siniestralidad y condiciones de riesgo constante.
  * Distribución Poisson y distribución Binomial Negativa (parametrización actuarial).
  * Procesos Poisson compuestos y procesos de saltos puros.
  * Procesos de contagio y mezclas estocásticas (Procesos de Pólya).
  * Frecuencia con probabilidades básicas variables en el tiempo.

### Track 4: Severidad, Pérdida Agregada y Simulación (`therisktheory.severity_aggregate`)
* **Alcance:**
  * Distribuciones continuas de cuantía de reclamo $X$: Lognormal, Pareto, Weibull, Beta.
  * Modelo individual vs. modelo colectivo de siniestralidad acumulada $S = \sum_{i=1}^N X_i$.
  * Función generadora de momentos, función característica y momentos agregados.
  * Métodos de aproximación analítica:
    * Aproximación Normal y corrección por asimetría (Edgeworth, Normal Power / NP).
    * Aproximación Gamma, transformaciones de Wilson-Hilferty y Haldane.
    * Transformada de Esscher para colas pesadas.
  * Simulación estocástica Monte Carlo vectorizada para agregación de riesgos.
  * Medidas de estabilidad y esquemas de reaseguro óptimo.

### Track 5: Teoría de la Credibilidad y Reservas (`therisktheory.credibility`)
* **Alcance:**
  * Prima de credibilidad lineal: $P_C = Z \cdot \bar{X} + (1 - Z) \cdot \mu$.
  * Credibilidad clásica total (criterio de exactitud y probabilidad) y credibilidad parcial.
  * Ponderadores de credibilidad $Z$ y propiedades estadísticas.
  * Credibilidad Bayesiana conjugada (Gamma-Poisson, Normal-Normal).
  * Métodos de credibilidad no paramétricos y semiparamétricos (Bühlmann y Bühlmann-Straub).
  * Modelos actuariales para reservas de siniestros ocurridos y no reportados (IBNR).

### Track 6: Teoría Colectiva del Riesgo (`therisktheory.collective_risk`)
* **Alcance:**
  * Proceso de riesgo colectivo en tiempo continuo y discreto: $U(t) = u + c t - S(t)$.
  * Probabilidad de insolvencia multiperíodo y probabilidad general de ruina $\psi(u)$.
  * Ecuación fundamental de Lundberg: $\mathbb{E}[e^{R(X - c)}] = 1$.
  * Coeficiente de ajuste $R$ y su cálculo numérico mediante `scipy.optimize`.
  * Cotas analíticas de ruina (Cota superior de Lundberg $\psi(u) \le e^{-Ru}$).
  * Efectos del reaseguro cuota parte y exceso de pérdida (XL) en el coeficiente de ajuste.

---

## 3. Plan de Desarrollo Semestral (16 Semanas)

### 📌 Resumen de Fases y Cronograma

```text
SEMANAS 1-4   ──>  FASE 1: Setup, Tipado Base y Funciones Matemáticas Fundamentales
SEMANAS 5-8   ──>  FASE 2: Modelos Nucleares, Convoluciones y Evaluación de Medio Término
SEMANAS 9-12  ──>  FASE 3: Transformaciones Avanzadas, Simulación Monte Carlo y Estabilidad
SEMANAS 13-16 ──>  FASE 4: Adaptadores de Backends, Validación Humana Completa y Release v0.2.0
```

### Detalle Semana a Semana

| Semana | Fase | Objetivos Clave y Entregables | Responsable / Track |
|:---:|:---:|---|:---:|
| **Sem 1** | Fase 1 | Configuración de entorno local (`uv`, Python 3.12, `pytest`), asignación de tracks. | Todos los Tracks |
| **Sem 2** | Fase 1 | Esqueletos de submódulos, contratos de interfaces, `pure_premium` ampliada y Poisson base. | Track 1 & Track 3 |
| **Sem 3** | Fase 1 | Ruina del jugador discreta, distribuciones Lognormal/Pareto y credibilidad total/parcial. | Track 2, Track 4, Track 5 |
| **Sem 4** | Fase 1 | **Hito 1 (Review 1):** Primeros PRs cruzados, cobertura de tests $\ge 75\%$, validación experimental. | Todos los Tracks |
| **Sem 5** | Fase 2 | Procesos Poisson compuestos, distribución Binomial Negativa y modelo colectivo individual. | Track 3 & Track 4 |
| **Sem 6** | Fase 2 | Teorema de De Finetti, modelo de Bühlmann y resolución numérica del coeficiente de Lundberg $R$. | Track 2, Track 5, Track 6 |
| **Sem 7** | Fase 2 | Reaseguro proporcional/no proporcional y aproximaciones Normal / Normal Power (NP). | Track 1 & Track 4 |
| **Sem 8** | Fase 2 | **Hito 2 (Midterm Evaluation):** Notebooks pedagógicos en `notebooks/`, cobertura $\ge 85\%$. | Todos / `@luispitta` |
| **Sem 9** | Fase 3 | Aproximación Gamma, transformaciones de Wilson-Hilferty, Haldane y Esscher. Cotas de Lundberg. | Track 4 & Track 6 |
| **Sem 10** | Fase 3 | Credibilidad Bühlmann-Straub, reservas IBNR y retención óptima bajo De Finetti. | Track 5 & Track 2 |
| **Sem 11** | Fase 3 | Generadores estocásticos vectorizados Monte Carlo y análisis de estabilidad de carteras. | Track 3, Track 4, Track 1 |
| **Sem 12** | Fase 3 | **Hito 3 (Review 3):** Congelamiento de API matemática y revisión exhaustiva de documentación. | Todos los Tracks |
| **Sem 13** | Fase 4 | Implementación de adaptadores para `pandas` y `Polars` con paridad numérica validada. | Capa de Backends |
| **Sem 14** | Fase 4 | Verificación de fuentes bibliográficas formales en `references/` y docstrings. | Todos los Tracks |
| **Sem 15** | Fase 4 | **Auditoría Matemática Integral:** Promoción a `mathematically_validated` con el instructor. | `@luispitta` |
| **Sem 16** | Fase 4 | **Hito 4 (Final Release v0.2.0):** Publicación del paquete, etiquetado de versión y cierre. | Todos / Mantenedor |

---

## 4. Ciclo de Vida de Validación Matemática

Cada fórmula o teorema incorporado al repositorio debe seguir una progresión formal de estados:

```text
[ CÓDIGO NUEVO + TESTS ]
          │
          ▼
    ┌────────────┐
    │experimental│  Funcionalidad implementada pero sin revisión matemática externa.
    └─────┬──────┘
          │  Revisión por pares (otro estudiante del curso)
          ▼
    ┌────────────┐
    │  reviewed  │  Derivación analítica, código y tests revisados por un par.
    └─────┬──────┘
          │  Aprobación formal definitiva del instructor / mantenedor
          ▼
┌──────────────────────────┐
│ mathematically_validated │  Fórmula, supuestos, referencias y límites certificados.
└──────────────────────────┘
```

> ⚠️ **Regla Inquebrantable:** Nadie puede validar matemáticamente su propia contribución. La aprobación final de `mathematically_validated` es potestad exclusiva de `@luispitta`.

---

## 5. Definición de Hecho (Definition of Done — DoD)

Para que un Pull Request sea aceptado y fusionado en `master`, debe satisfacer el 100% de esta lista:

- [ ] **Definición Matemática Rigurosa:** Ecuación teórica explicada en el docstring y notas técnicas.
- [ ] **Dominio y Supuestos Explícitos:** Validación defensiva de parámetros (ej. $\lambda > 0$, $\alpha > 0$).
- [ ] **Vectorización Completa:** Implementación usando NumPy/SciPy sin bucles Python en operaciones matemáticas.
- [ ] **Anotación de Tipos (PEP 484/561):** Tipado estricto en argumentos y valor de retorno.
- [ ] **Docstrings Estándar (Google-style, 100% en inglés):** Con secciones `Args`, `Returns`, `Raises`, `Notes`, `References` y `Examples`.
- [ ] **Idioma Inglés Estricto:** Código, comentarios, mensajes de excepción y tests redactados exclusivamente en inglés.
- [ ] **Cita Bibliográfica Exacta:** Referencia bibliográfica verificable (autor, libro, capítulo o paper).
- [ ] **Tests Analíticos:** Validación contra casos cerrados conocidos y tablas actuariales estándar.
- [ ] **Tests de Vectores y Broadcasting:** Comprobación de que la función acepta arreglos multidimensionales.
- [ ] **Tests de Límites y Errores:** Validación de que se lancen excepciones adecuadas ante entradas inválidas.
- [ ] **Ejemplo Ejecutable:** Script o notebook funcional de demostración.
- [ ] **Revisión por Pares Aprobada:** Feedback y aprobación por otro miembro del equipo.
- [ ] **Aprobación de Merge:** Aprobación formal otorgada por `@luispitta`.

---

## 6. Estándares Técnicos y Reglas de Rendimiento

### 6.1 Buenas Prácticas de Vectorización

```python
# FORMA CORRECTA (Vectorizada, O(1) en Python, ejecución en C/BLAS)
import numpy as np


def pure_premium(frequency: np.ndarray, severity: np.ndarray) -> np.ndarray:
    freq = np.asarray(frequency, dtype=np.float64)
    sev = np.asarray(severity, dtype=np.float64)
    if np.any(freq < 0) or np.any(sev < 0):
        raise ValueError("Frequency and severity must be non-negative.")
    return freq * sev
```

```python
# FORMA PROHIBIDA (Iteración fila a fila, ineficiente y no actuarial)
def pure_premium_bad(dataframe):
    results = []
    for row in dataframe.iterrows():  # ¡PROHIBIDO!
        results.append(row["freq"] * row["sev"])
    return results
```

### 6.2 Comandos Esenciales de Verificación

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Ejecutar tests con reporte de cobertura
pytest --cov=therisktheory --cov-report=term-missing

# Ejecutar verificación de estilo y linter
ruff check .

# Formatear código según los estándares del proyecto
ruff format .
```
