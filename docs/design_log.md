# TALOS · Design Log

> Bitácora de decisiones de diseño de la UI envolvente de TALOS.
> La lógica interna de los módulos (`Circuit2d`, `SpeedHeatMap`, `Telemetrytrace`,
> `TeamTelemetry`, etc.) **no** se ha modificado. Sólo se ha reorganizado el
> shell que los aloja.

---

## 2026-04-26 · v0.9 — Rediseño "Mission Control"

> **Shell objetivo**: `src/appResearch.py` — es el enrutador real (con todos
> los módulos cableados). `src/app.py` es el esqueleto inicial con
> placeholders y se ha dejado tal cual; **no es el shell vivo**.

### Contexto

El esqueleto previo de `appResearch.py` era funcional pero dependía al 100 %
de la barra lateral de Streamlit:

- Toda la navegación (categorías + módulos) vivía en el sidebar.
- La configuración de sesión (año, GP, tipo) era un formulario plano al pie.
- El "hub" era una rejilla de botones simples sin jerarquía visual ni
  identidad propia.

El objetivo de esta iteración es **transformar el shell** para que se sienta
como un programa profesional de telemetría — referencia visual: **McLaren
ATLAS**, **SAP Fiori** y el portal interno de la **FIA** — sin tocar la
lógica de los módulos.

### Decisiones

#### 1 · Arquitectura de tres modos

El enrutador principal pasa a tener **tres estados** mutuamente excluyentes:

| Estado            | Trigger                                    | Render                          |
|-------------------|--------------------------------------------|---------------------------------|
| Mission Control   | `session_loaded == False`                  | Selector de sesión a pantalla completa |
| Cockpit (Hub)     | `session_loaded == True && active_module is None` | Tarjeta de sesión + lanzador 2×2 de módulos |
| Module View       | `active_module is not None`                | Topbar + command bar + module switcher + módulo |

El sidebar deja de ser navegación y queda **minimalizado** — sólo conserva
estado de sesión, info de build y dos atajos de emergencia (volver al Hub /
cambiar sesión). Streamlit arranca con el sidebar **colapsado por defecto**
(`initial_sidebar_state="collapsed"`).

#### 2 · Mission Control · selector tipo "FIA portal"

El usuario configura la sesión en **tres pasos numerados** con encabezados
estilo "STEP 01 / SEASON":

- **Step 01 — Season**: chips horizontales 2018‒2025. Año activo se renderiza
  como botón `type="primary"` (rojo F1). Cambiar de año invalida el GP
  seleccionado, ya que el calendario varía.
- **Step 02 — Session Type**: chips FP1 · FP2 · FP3 · Q · S · R, mismo patrón
  visual.
- **Step 03 — Grand Prix**: rejilla 4×N de **tarjetas de GP** con
  bandera del país (emoji), número de ronda, nombre y localidad. La tarjeta
  seleccionada cambia su borde superior a rojo. Cada tarjeta es un bloque
  HTML decorativo + un `st.button` adyacente que actúa como gatillo.

El calendario se obtiene vía `fastf1.get_event_schedule(year)` con
`@st.cache_data` — se incluye un calendario **estático de fallback** (24 GPs)
para el caso de fallo de red.

CTA final centrado: `INITIALIZE SESSION`. Mientras no haya GP seleccionado,
muestra un botón deshabilitado con copy explícito.

#### 3 · Topbar fija estilo SAP Shell Bar

Banner HTML/CSS al inicio de cada vista con tres zonas:

- **Izquierda**: marca cuadrada roja (◇) + wordmark `TALOS` (Share Tech Mono,
  letterspacing 8 px) + tagline + breadcrumb dinámico
  (`HUB › COCKPIT`, `HUB › ANÁLISIS DE PILOTO › CIRCUITO ANIMADO`, …).
- **Derecha**: pulse-indicator del estado de la sesión (verde "live" cuando
  hay sesión cargada, amarillo "idle" cuando no), versión del build.

El componente es **decorativo** — los botones interactivos viven en la
**command bar** justo debajo (limitación de Streamlit: no se puede inyectar
HTML con widgets dentro). La separación funciona bien visualmente.

#### 4 · Command bar bajo el topbar (vista de módulo)

Tres botones cortos:

- `◀ HUB` — vuelve al Cockpit conservando la sesión.
- `⟲ SESIÓN` — descarta sesión y vuelve a Mission Control.
- Toggle `Curvas` (sinónimo del antiguo `corners` del sidebar).

Debajo, un **module switcher** horizontal que muestra las hermanas de la
categoría activa como pestañas, marcando la activa con `type="primary"`.
Esto reemplaza la dependencia del sidebar para saltar entre módulos
relacionados sin perder contexto.

#### 5 · Cockpit (post-sesión)

Cuando hay sesión cargada pero no se ha elegido módulo:

- **Session card** prominente con bandera grande, nombre del GP, año, tipo
  de sesión, número de pilotos y total de vueltas + tres pills
  decorativas (`◉ LIVE FEED`, `FASTF1`, `R/Q/FP*`).
- Tira de **4 métricas** (pilotos, vueltas, tipo, año) usando los
  `st.metric` de Streamlit con borde rojo superior — coherente con el
  estilo "telemetry panel" de ATLAS.
- **Module Launcher** — quadrante 2×2 donde cada panel es una categoría
  con icono geométrico (◷ ◉ ⇄ ◬), título, línea roja decorativa,
  descripción y los botones de los módulos de esa categoría.

#### 6 · Status bar inferior

Pie de página simulado tipo "barra de estado" de SAP — muestra: estado de
sync, año, sesión, GP y origen de datos (`FASTF1 · ERGAST · TALOS · 14
MODULES`). Refuerza la sensación de "programa de escritorio" frente a
"dashboard web".

#### 7 · Sistema visual

- **Tipografía**: dos familias importadas (`Titillium Web` para texto largo,
  `Share Tech Mono` para datos, etiquetas y dígitos) + `JetBrains Mono`
  como reserva.
- **Paleta**: definida vía variables CSS en `:root` (`--bg-0..4`, `--accent`,
  `--pit-green`, `--warn-yellow`, `--info-blue`, `--gold`/`--silver`/
  `--bronze`).  Centralización para mantenimiento.
- **Geometría**: bordes 2 px, esquinas casi rectas (`border-radius: 2px`),
  acentos rojos en bordes superiores e izquierdos. Sin sombras suaves; las
  pocas que hay son redondas y rojas (glow del logo, gradientes radiales
  en el fondo de la app y en el banner de Mission Control).
- **Pasos numerados** + **breadcrumbs** + **status bar**: trío recurrente
  en software industrial que da sensación de proceso/flujo controlado.

#### 8 · Cosas que se han descartado

- **Tabs nativos de Streamlit (`st.tabs`)** para la navegación: pierden el
  estado al rerender y no permiten estilarlos como pestañas industriales.
  Se sustituye por una fila de `st.button(type=…)` controlada por
  `session_state`.
- **`st.pills` / `st.segmented_control`**: aunque están disponibles en
  1.40+ y aquí tenemos 1.54, su estilado es limitado. Los chips
  artesanales con `st.button` son más controlables.
- **Header HTML con botón clicable**: `st.button` no acepta hijos HTML;
  para no fragmentar, se renderiza el "card" como `<div>` decorativo y el
  botón se coloca debajo. Funciona y mantiene la accesibilidad nativa.

### Archivos tocados

- `src/appResearch.py` — **shell reescrito completo**. Se preserva
  literalmente cada `elif active == "…"` con su llamada a
  `render_*` original (`ss.render_session_summary`, `stl.render_timeline`,
  `sgrid.render_strategy_dashboard`, `sm.render_speed_heatmap`,
  `gm.render_gear_heatmap`, `trace.render_telemetry_trace`,
  `tteam.render_team_telemetry`, `os.render_session_compare`,
  `ol.render_ideal_lap`, `ghost.render_ghost_car`, `gg.render_demand_map`,
  `aero.render_aero_analysis`, `tload.render_tyre_load`). También se
  conserva el helper `select_driver_and_lap` y la tarjeta de piloto Ergast
  embebida en "Telemetry Trace".
- `src/assets/styles.css` — reescrito (topbar, mc-banner, gp-card, cat-panel,
  ses-card, statusbar, command bar, chips, refinos de inputs/alertas/métricas).
- `src/app.py` — **sin cambios** (es el esqueleto antiguo con placeholders;
  no es el shell que se ejecuta).
- `src/.streamlit/config.toml` — sin cambios.
- `docs/design_log.md` — este archivo.

### Cómo arrancar

```bash
cd src
streamlit run appResearch.py
```

(El otro `app.py` existe pero corresponde al esqueleto antiguo con
placeholders; no debe usarse como shell.)

---

## 2026-04-27 · v0.10 — Categoría "Resúmenes" (insights derivados)

### Contexto

Se introduce una nueva categoría **al inicio del NAV** (`Resúmenes`) con dos
módulos analíticos derivados de la telemetría — ambos diseñados para ser
**100 % deterministas y validables**, requisito explícito del TFG. No se
asume nada que no se pueda extraer directamente de FastF1.

### Módulos añadidos

#### `CircuitSummary.py` — Clasificación de circuitos

Clasifica el circuito de la sesión activa en una de cinco tipologías
mediante un sistema de **reglas determinísticas** sobre features extraídos
de la vuelta más rápida:

| Tipología   | Criterio dominante                                |
|-------------|---------------------------------------------------|
| POWER       | `% gas a fondo ≥ 65` y `vel.media ≥ 215 km/h`      |
| HIGH-SPEED  | `G lat. media ≥ 1.6 G` y/o `G lat. máx ≥ 4.5 G`    |
| STOP-AND-GO | `% freno ≥ 28` y/o `frenada ≥ 5.0 G`               |
| STREET      | `vel.media < 170` y `% gas a fondo < 45`           |
| TECHNICAL   | fallback (parte con +1 base, gana sin extremos)    |

**Cálculo de fuerzas G** desde telemetría pura:
- `g_lon = dV/dt / 9.81`
- `g_lat = V · dθ/dt / 9.81` con `θ = arctan2(dY, dX)` desempaquetado
- Suavizado con media móvil de 5 muestras

**Visualizaciones**:
- Tarjeta principal con icono geométrico, color del tipo y descripción
- 12 métricas (velocidad, throttle, freno, G, longitud, mejor vuelta…)
- Radar comparativo perfil_actual vs 5 perfiles de referencia
- Mapa del trazado coloreado por G total instantánea
- Perfil de velocidad vs distancia
- Lista de reglas activadas (transparencia total del clasificador)

**Validación manual**: ejecutar sobre Mónaco → STREET, Monza → POWER,
Suzuka → HIGH-SPEED, Singapur → STOP-AND-GO, Barcelona → TECHNICAL.

#### `DriverSummary.py` — Clustering de estilos de conducción

Agrupa a los pilotos de la sesión activa con **K-Means** (`random_state=42`,
`n_init=10` — completamente determinista) sobre 10 features extraídos de la
mejor vuelta de cada piloto:

```
speed_avg · speed_max · speed_std
throttle_full_pct · throttle_avg
brake_pct · coast_pct
gear_change_rate · drs_usage_pct
lap_consistency  (= exp(-std(LapTime)) → 0..1)
```

**Pipeline**: `StandardScaler` → `KMeans` → `PCA(2)` para visualización +
heurística de etiquetado automático (CONSISTENTE / AGRESIVO / VELOZ-EQUILIBRADO
/ CONSERVADOR / ATACANTE / INTERMEDIO).

**Validación triple**:
1. **Matemática · Silhouette score** — métrica visible en la cabecera; se
   muestra delta "bueno (>0.4)" / "débil (<0.4)".
2. **Geométrica · curva Silhouette para K=2..6** — ayuda a justificar la
   K elegida con la línea verde sobre el K óptimo.
3. **Deportiva · correlación de Spearman** entre el rank del cluster
   (ordenado por velocidad media descendente) y la posición final del
   piloto en la sesión. Sólo aplica si la sesión es R/Q con resultados.

**Visualizaciones**:
- Métrica grande de Silhouette
- PCA 2D scatter (cada piloto coloreado por cluster)
- Radar de centroides normalizados
- Curva de Silhouette para diferentes K
- Tabla con asignación, equipo, "feature destacado" (z-score máximo)
- Bloques de "insights deportivos" con la correlación Spearman interpretada

### NAV reorganizado

```
◆ Resúmenes              ← NUEVA
  · Resumen del Circuito (CircuitSummary)
  · Resumen del Piloto   (DriverSummary)
◷ Información de Sesión
  · Resumen de la sesión, Cronología, Estrategia
◉ Análisis de Piloto (Micro)
⇄ Comparativas Competitivas
◬ Dinámica Vehicular
```

El cockpit del Hub se generaliza para mostrar dinámicamente cualquier
número de categorías en grid 2-columnas (antes hardcoded a 2×2).

### Determinismo / Validabilidad

- Ningún módulo nuevo asume datos externos a FastF1.
- Ningún módulo nuevo introduce aleatoriedad (semillas fijas o nada).
- Todas las decisiones del clasificador de circuitos son rastreables — la
  pestaña *"Reglas aplicadas"* lista cada disparo con el valor que lo activó.
- El clustering de pilotos reporta la métrica matemática (Silhouette), la
  curva de optimización (mejor K) y la validación deportiva (Spearman)
  en la propia UI.

### Archivos tocados

- `src/CircuitSummary.py` — **nuevo** (~440 líneas).
- `src/DriverSummary.py` — **nuevo** (~430 líneas).
- `src/appResearch.py` — `NAV` y `CATEGORY_META` extendidos; cockpit
  generalizado; dispatcher con dos nuevas ramas.
- Resto sin tocar.

### Próximos pasos sugeridos

- Cablear los módulos ya implementados pero aún en placeholder
  (`SessionTimeLine`, `Strat_Grid`, `GGDiagram`, `Aero`, `Head2head`,
  `OptimalLap`, `Ghostcar`, `TyreLoad`, `OverallStandings`,
  `SessionSummary`) al dispatcher de `app.py`.
- Añadir una **vista "Race Summary"** en el Cockpit que embeba el módulo
  `SessionSummary.render_session_summary` para dar contexto antes de
  saltar a un análisis específico.
- Persistir el último GP seleccionado entre runs vía `st.cache_resource`
  o un `~/.talos.toml` para evitar que el usuario tenga que reseleccionar
  cada vez que reinicia la app.
- Considerar introducir una vista **"Driver Dossier"** (tarjeta del piloto
  con stats Ergast) reutilizable, ya que actualmente vive incrustada en
  el bloque de "Circuito Animado".
