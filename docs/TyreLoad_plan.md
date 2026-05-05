# TyreLoad.py — Plan de Implementación

## Módulo: Mapa de Carga por Rueda (Tire Stress Map)

**Objetivo:** Visualizar la distribución de carga dinámica en las 4 ruedas del monoplaza
a lo largo de una vuelta, usando únicamente G_lat y G_lon de la telemetría FastF1.

---

## 1. Modelo Físico — Distribución de Carga por Rueda

### 1.1 Concepto

Un F1 genera fuerzas G laterales (curvas) y longitudinales (frenada/aceleración).
Estas fuerzas transfieren peso entre las ruedas:

- **G_lat positivo** (curva izquierda) → carga se transfiere a ruedas **derechas**
- **G_lat negativo** (curva derecha) → carga se transfiere a ruedas **izquierdas**
- **G_lon positivo** (aceleración) → carga se transfiere al **eje trasero**
- **G_lon negativo** (frenada) → carga se transfiere al **eje delantero**

### 1.2 Fórmulas de Distribución

Usamos un modelo simplificado de transferencia de carga normalizado (0–1):

```
Base estática = 0.25 por rueda (distribución uniforme)

Transferencia lateral (normalizada):
  lat_n = G_lat / P98(|G_lat|)   → rango aprox [-1, 1]

Transferencia longitudinal (normalizada):
  lon_n = G_lon / P98(|G_lon|)   → rango aprox [-1, 1]

Carga por rueda (antes de normalización final):
  FL = 0.25 - 0.25·lon_n + 0.25·lat_n    (frontal izquierdo)
  FR = 0.25 - 0.25·lon_n - 0.25·lat_n    (frontal derecho)
  RL = 0.25 + 0.25·lon_n + 0.25·lat_n    (trasero izquierdo)
  RR = 0.25 + 0.25·lon_n - 0.25·lat_n    (trasero derecho)

Normalización final por fila:
  load_i = clamp(load_i, 0, 1)
  load_i = load_i / max(FL, FR, RL, RR)   → la rueda más cargada = 1.0
```

Convención de signos:
- `lon_n < 0` → frenada → carga va al eje delantero → FL,FR suben
- `lon_n > 0` → aceleración → carga va al eje trasero → RL,RR suben
- `lat_n > 0` → curva izquierda → carga va a la derecha → FR,RR suben
- `lat_n < 0` → curva derecha → carga va a la izquierda → FL,RL suben

> **Nota:** Este modelo es determinista y no requiere parámetros del vehículo.
> Es una aproximación cualitativa que muestra la *dirección* de la transferencia,
> no la magnitud exacta en Newtons.

### 1.3 Fuente de Datos

- Reutilizar `_calculate_g_forces()` del patrón establecido en GGDiagram/Aero
- Canales requeridos: `Time`, `X`, `Y`, `Speed`, `Distance`
- Salida: `g_lat`, `g_lon` (filtrados con Savitzky-Golay, clip ±6G)

---

## 2. Visualizaciones

### 2.1 Track Map Animado (Plotly con Frames)

- **Base:** Trazado del circuito en gris (patrón Circuit2d.py)
- **Cursor animado:** Punto que recorre el circuito
- **Color del trail:** Escala de G_total (verde→amarillo→rojo)
- **Controles:** Play/Pause + Slider de distancia
- **Sincronizado** con la silueta F1

### 2.2 Silueta F1 Top-View con 4 Neumáticos

- Renderizado con **Plotly shapes** (rectángulos, paths SVG simplificados)
- Silueta esquemática del monoplaza vista cenital
- **4 rectángulos** representando neumáticos en FL, FR, RL, RR
- **Color:** Escala verde (#39FF14) → amarillo (#FFD600) → rojo (#E8002D)
- Se actualiza en cada frame de la animación

### 2.3 Métricas KPI (st.metric)

| Métrica | Descripción |
|---------|-------------|
| FL Max Load | Máximo normalizado front-left |
| FR Max Load | Máximo normalizado front-right |
| RL Max Load | Máximo normalizado rear-left |
| RR Max Load | Máximo normalizado rear-right |
| Avg Total Load | Media de las 4 ruedas |
| Peak G Total | G total máximo registrado |

### 2.4 Gráfico de Carga vs Distancia

- 4 líneas (FL, FR, RL, RR) sobre eje X = distancia
- Colores: cyan (FL), amber (FR), green (RL), purple (RR)
- Fill-to-zero con opacidad baja
- Patrón de `_build_g_vs_distance()` en GGDiagram.py

---

## 3. Arquitectura del Módulo

### 3.1 Funciones

```
TyreLoad.py
│
├── CONSTANTES (colores, font, escalas)
├── _calculate_g_forces(tel)           # Reutilizar patrón GGDiagram/Aero
├── _compute_wheel_loads(tel)          # NUEVO: Modelo de transferencia
├── _load_to_color(value: float)       # NUEVO: Valor 0-1 → hex verde→rojo
├── _build_car_silhouette(fl,fr,rl,rr) # NUEVO: Silueta F1 top-view
├── _build_animated_figure(...)        # NUEVO: Track + Silueta sincronizados
├── _build_load_vs_distance(tel)       # NUEVO: 4 canales de carga
├── _render_load_metrics(tel)          # NUEVO: KPIs
├── _apply_dark(fig, height)           # Helper reutilizado
└── render_tyre_load(session, drv, lap) # PUNTO DE ENTRADA PÚBLICO
```

### 3.2 Integración en app.py

**NAV** — añadir en "Dinámica Vehicular":
```python
"Dinámica Vehicular": [
    "Diagrama G-G",
    "Carga Aerodinámica",
    "Mapa de Carga por Rueda",    # ← NUEVO
    "Modelo Térmico de Frenos",
],
```

**Enrutador** — añadir elif con selector piloto+vuelta (patrón estándar).

---

## 4. Layout Visual

```
┌─────────────────────────────────────────────────┐
│  HEADER: "MAPA DE CARGA POR RUEDA"             │
├─────────────────────────────────────────────────┤
│  KPIs: [FL Max] [FR Max] [RL Max] [RR Max] ... │
├──────────────────────┬──────────────────────────┤
│   TRACK MAP          │   SILUETA F1 TOP-VIEW    │
│   (animado, 65%)     │   con 4 neumáticos (35%) │
│                      │   coloreados              │
│   ● cursor           │   ┌──┐      ┌──┐         │
│   ══ trail color     │   │FL│      │FR│         │
│                      │   └──┘  ██  └──┘         │
│                      │   ┌──┐      ┌──┐         │
│                      │   │RL│      │RR│         │
│                      │   └──┘      └──┘         │
├──────────────────────┴──────────────────────────┤
│  [▶ PLAY] [⏸ PAUSE]  ═══════ slider            │
├─────────────────────────────────────────────────┤
│  CARGA POR RUEDA vs DISTANCIA (4 líneas)        │
└─────────────────────────────────────────────────┘
```

---

## 5. Plan de Ejecución

| Paso | Descripción | Dependencias |
|------|-------------|-------------|
| 1 | Crear `TyreLoad.py` — esqueleto, constantes, imports | Ninguna |
| 2 | `_calculate_g_forces()` — copiar de Aero.py | Paso 1 |
| 3 | `_compute_wheel_loads()` — modelo de transferencia | Paso 2 |
| 4 | `_load_to_color()` — interpolación verde→rojo | Paso 1 |
| 5 | `_build_car_silhouette()` — silueta F1 con Plotly | Paso 4 |
| 6 | `_build_animated_figure()` — track map + silueta con frames | Pasos 3,5 |
| 7 | `_build_load_vs_distance()` + `_render_load_metrics()` | Paso 3 |
| 8 | `render_tyre_load()` + integración en `app.py` | Pasos 6,7 |

---

## 6. Decisiones de Diseño

| Decisión | Justificación |
|----------|---------------|
| Sin parámetros del vehículo | Consistencia con filosofía TALOS: solo telemetría pública |
| Normalización 0–1 | Comparación cualitativa sin unidades absolutas |
| Reutilizar `_calculate_g_forces` | Consistencia con GGDiagram y Aero |
| Plotly shapes para silueta | No depende de assets, vectorial, animable |
| Escala verde→rojo | Intuitiva: verde = poca carga, rojo = máxima |
| Frames de Plotly | Patrón probado en Circuit2d, máxima fluidez |
| P98 para normalización | Evita que un outlier aislado aplaste toda la escala |

---

## 7. Limitaciones Declaradas

- Distribución **cualitativa**, no cuantitativa
- Sin masa, altura CG, ancho de vía y batalla → no es transferencia real en N
- Asume distribución estática 50/50 F/R (F1 real es ~45/55)
- No incluye downforce variable con velocidad (eso está en Aero.py)
- Silueta F1 esquemática, no a escala real
