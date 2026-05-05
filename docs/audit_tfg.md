# Informe de Auditoría Técnica TFG: Sistema de Telemetría F1 (TALOS)

**Fecha de Auditoría:** Abril 2026
**Rol:** Auditor Senior de Software / Evaluador de Tribunal
**Estado General:** Alerta Naranja (Riesgo arquitectónico y documental alto, potencial técnico excelente).

---

## 1. Reality Check (Viabilidad y Tiempos)

**¿Es suficiente el volumen técnico de FastF1 + Lógica Difusa para aprobar un TFG en Ingeniería del Software?**
**Sí, absolutamente.** Pero hay un matiz crítico: el tribunal no evalúa si eres un buen analista de Fórmula 1, evalúa si eres un buen *Ingeniero de Software*.
Actualmente tienes un proyecto con un potencial analítico altísimo (procesamiento de señales, DataFrames masivos, visualizaciones polares), pero **empaquetado como un script de prototipado de Data Science**, no como un producto de ingeniería. 

Si te presentas con la documentación "en pañales" y un código espagueti monolítico, un tribunal estricto te suspenderá o te bajará drásticamente la nota por falta de rigor metodológico, falta de pruebas automatizadas y ausencia de patrones de diseño. Tienes el núcleo duro hecho; ahora toca vestirlo de ingeniería.

---

## 2. Puntos Fuertes (Lo Defendible)

Si el tribunal ataca, estos son tus escudos:

1. **Procesamiento de Señales Reales:** La implementación del filtro de Savitzky-Golay con ventana adaptativa en `GGDiagram.py` para limpiar el ruido de cuantización del GPS es una justificación técnica brillante de nivel de ingeniería.
2. **Desarrollo de Motores de Inferencia Propios:** No has usado una "caja negra" externa para la lógica difusa. Tu implementación basada en sigmoides (`_sigmoid_score`) y percentiles relativos a la sesión demuestra comprensión matemática profunda y adaptabilidad algorítmica.
3. **Optimización Vectorial:** En partes críticas (como el cálculo de fuerzas G en `GGDiagram.py`), has usado operaciones vectorizadas de `numpy` (`np.diff`, `np.gradient`, `np.unwrap`) en lugar de iterar con bucles for. Esto es vital para procesar telemetría de 240Hz en tiempo real.
4. **Visualización Avanzada (UI/UX Analítico):** Los diagramas polares y la integración de gráficos coordinados en Plotly (trazados que comparten ejes X de distancia) denotan madurez en el entendimiento de interfaces de usuario para visualización de datos complejos.

---

## 3. Puntos Críticos (Riesgo de Suspenso)

Estos son los vectores de ataque garantizados por parte del tribunal en la defensa:

*   **Deuda Técnica de UI (El "CSS Espagueti"):** En `app.py` tienes bloques de `<style>` inyectados a machete mediante `st.markdown(..., unsafe_allow_html=True)`. Un tribunal de software verá esto como una "chapuza". **Solución:** Mueve los estilos a `.streamlit/config.toml` o carga un archivo `style.css` externo estático.
*   **Acoplamiento y Falta de Modularidad (Fuzzy Logic Duplicada):** Tienes la lógica de los motores difusos hardcodeada dentro de los módulos de presentación (`Telemetrytrace.py` y `GGDiagram.py`). Si quieres cambiar la constante `center=3.0` de una sigmoide, tienes que bucear en código de UI.
*   **Cuellos de Botella de Rendimiento (Bloqueo del Hilo Principal):** Al cargar una sesión de FastF1 (`fastf1.get_session().load()`), el hilo principal de Streamlit se bloquea completamente. Aunque usas `@st.cache_data`, la primera carga es síncrona. Si el tribunal te pregunta cómo escalarías esto a 100 usuarios concurrentes, tu arquitectura actual colapsaría.
*   **Antipatrones de Pandas:** En algunas partes usas `iterrows()` (ej. `for _, lap in laps.iterrows():` en la baseline). Esto es un pecado mortal en el manejo de DataFrames en Python.
*   **Ausencia Total de Pruebas (Tests):** Presentar un TFG de Ingeniería del Software sin tests unitarios es un suspenso inminente con un evaluador estricto. ¿Cómo garantizas que el cálculo de `_extract_pedal_stats` es correcto si mañana actualizas Pandas?

---

## 4. Triaje de Tareas (Priorización Implacable)

Tienes el tiempo en contra. Hay que aplicar la regla de Pareto (80/20).

### ❌ Lo que sobra (ABANDONAR INMEDIATAMENTE)
1. **Módulo de simulación post-sesión (subida de datos):** Un sumidero de horas. Descártalo. Argumenta en la memoria que se ha dejado como "Trabajo Futuro".
2. **Telemetría UDP Live:** Tíralo. Demasiado complejo lidiar con sockets asíncronos en Streamlit y problemas de concurrencia a semanas de la entrega. Céntrate en hacer perfecto el análisis histórico.
3. **Refactorizaciones "puristas" masivas:** No intentes pasar de Streamlit a React+FastAPI ahora. Morirás.

### ✅ Lo que falta (MANDATORIO)
1. **Refactor de Motores Difusos:** Extrae las funciones `_sigmoid_score`, `_compute_session_baseline` y la lógica de inferencia a un fichero `src/fuzzy_inference.py`. Desacopla la lógica de negocio de la vista.
2. **Dockerización (Tu salvavidas):** Un `Dockerfile` y un `docker-compose.yml` sencillos. El tribunal adora Docker. Justifica que garantiza la "reproducibilidad del entorno" (FastF1 tiene dependencias de C++ que pueden fallar en Windows/Mac, Docker lo soluciona).
3. **Testing de la Lógica de Negocio:** Añade `pytest` a tus `requirements.txt`. Escribe tests **únicamente** para las funciones de cálculo matemático y lógica difusa (no testees la UI de Streamlit). Unos 10-15 tests te salvarán la vida.

---

## 5. Estrategia Documental

Tu memoria debe centrarse en vender el producto no como "una web que pinta gráficas", sino como un **Sistema de Soporte a la Decisión (DSS - Decision Support System) basado en procesamiento masivo de datos**.

*   **Ingeniería Inversa de los Mock-ups:** Si el anteproyecto tenía mock-ups que no cuadran con la realidad actual, **justifícalo metodológicamente**. Usa palabras como: *"Tras aplicar metodologías ágiles y prototipado iterativo, descubrimos que los usuarios (ingenieros) valoraban más la densidad de datos en la pantalla que el espacio en blanco de los diseños originales, pivotando hacia un paradigma de Data Terminal."*
*   **Análisis de Requisitos:**
    *   *Funcionales:* Carga de telemetría histórica, comparación superpuesta de trazadas, cálculo de derivadas (G-Force), generación de insights por lógica difusa.
    *   *No Funcionales:* Tiempo de respuesta (<2s tras el cacheo inicial), disponibilidad cross-platform (gracias a Docker), resiliencia a datos nulos de la FIA.
*   **Arquitectura:** Dibuja un diagrama (UML o similar) mostrando una arquitectura de "Capas limpias":
    *   Capa de Integración (FastF1 API)
    *   Capa de Lógica de Negocio (Motores Fuzzy, Procesamiento de Señales)
    *   Capa de Presentación (Streamlit / Plotly)

---

## 6. Plan de Choque (Roadmap hasta el depósito)

Asumiendo que tienes 4-6 semanas:

### Fase 1: Limpieza del Código (Días 1-4)
- **Objetivo:** Eliminar el código espagueti.
- Mover `<style>` a `.streamlit/config.toml` o a un `.css` cargado estáticamente.
- Refactorizar las funciones lógicas de `GGDiagram.py` y `Telemetrytrace.py` a un archivo centralizado (ej. `core/analytics.py` o `core/fuzzy.py`).

### Fase 2: Robustez (Días 5-8)
- **Objetivo:** Ingeniería de verdad.
- Crear `Dockerfile` basado en `python:3.12-slim`.
- Instalar `pytest` y crear el directorio `/tests`. Escribir tests unitarios para comprobar que las sigmoides y cálculos de G-Forces no explotan con valores nulos (`np.nan`).

### Fase 3: La Memoria - Ingeniería Inversa (Días 9-20)
- **Objetivo:** Redactar el documento.
- Hacer las capturas de pantalla de la app en modo oscuro y alta resolución.
- Escribir el Análisis de Requisitos copiando lo que ya hace el programa.
- Explicar la fórmula matemática de las G-Forces y la sigmoide difusa en la sección de "Diseño Técnico". Esto te sube 1 punto entero en la nota.

### Fase 4: Defensa y Presentación (Días 21+)
- **Objetivo:** Preparar el "Escudo".
- Hacer un PDF de presentación de máximo 15-20 diapositivas.
- Prepara una demo en vídeo de 1 minuto por si falla el directo el día de la defensa (Ley de Murphy).
- Preparar respuestas a las preguntas trampa: *"¿Por qué usas Streamlit y no una arquitectura de microservicios separando frontend y backend?"* -> *"Para un flujo de Data Science puro de este alcance, el acoplamiento vista-datos de Streamlit reduce el time-to-market sin sacrificar capacidad de procesamiento vectorial de Pandas."*
