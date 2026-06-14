import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Simulador Vacunatorio TP5", layout="wide")
st.title("💉 Simulador de Vacunatorio - UTN FRC")

# ==========================================
# MÓDULO 1: PARÁMETROS DEL USUARIO (Sidebar)
# ==========================================
st.sidebar.header("⚙️ Parámetros Discretos")

# Parámetros de Llegadas
media_llegada_covid = st.sidebar.number_input("Media Llegada COVID (min)", value=3.75, step=0.1) 
media_llegada_gripe = st.sidebar.number_input("Media Llegada Gripe (min)", value=5.0, step=0.1)

# Parámetros de Vacunación
tiempo_vacunacion_seg = st.sidebar.number_input("Tiempo de aplicación (seg/paciente)", value=22.0)

# Interrupción Inventada (Pausa de Sanitización)
st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Interrupción Inventada")
frecuencia_limpieza = st.sidebar.number_input("Frecuencia Limpieza (min)", value=60.0)
duracion_limpieza = st.sidebar.number_input("Duración Limpieza (min)", value=5.0)

# Parámetros de Visualización (Obligatorios del TP)
st.sidebar.markdown("---")
st.sidebar.subheader("👀 Visualización")
iteraciones_max = st.sidebar.number_input("Iteraciones Máximas", value=100000)
fila_j = st.sidebar.number_input("Mostrar desde fila (j)", value=0, min_value=0)
filas_i = st.sidebar.number_input("Cantidad de filas (i)", value=120, min_value=1)

# Parámetros Runge-Kutta
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros Continuos (RK)")
paso_h = st.sidebar.number_input("Paso de integración (h)", min_value=0.0001, max_value=1.0, value=0.1, step=0.01)
r_inicial = st.sidebar.number_input("Condición inicial R(0)", min_value=0.0, max_value=1.0, value=0.076, step=0.001)

# ==========================================
# MÓDULO 2: RUNGE-KUTTA 4° ORDEN
# ==========================================
@st.cache_data # Usamos caché para que no recalcule RK si no cambiamos el "h" o "R(0)"
def calcular_runge_kutta(h, r0):
    filas_rk = []
    t = 0.0
    r = r0
    
    filas_rk.append({"t": round(t, 4), "R": round(r, 6)})
    max_iteraciones = 100000 
    iteracion = 0
    
    while iteracion < max_iteraciones:
        derivada_actual = 41.4 * r - 0.0575 * (r ** 2)
        
        # Criterio de parada: cuando la derivada es casi cero (se estabilizó)
        if derivada_actual <= 0.00001 and iteracion > 0:
            break
            
        k1 = 41.4 * r - 0.0575 * (r ** 2)
        
        r_k2 = r + (h / 2.0) * k1
        k2 = 41.4 * r_k2 - 0.0575 * (r_k2 ** 2)
        
        r_k3 = r + (h / 2.0) * k2
        k3 = 41.4 * r_k3 - 0.0575 * (r_k3 ** 2)
        
        r_k4 = r + h * k3
        k4 = 41.4 * r_k4 - 0.0575 * (r_k4 ** 2)
        
        r_siguiente = r + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_siguiente = t + h
        
        if r_siguiente <= r:
            break
            
        t = t_siguiente
        r = r_siguiente
        iteracion += 1
        
        filas_rk.append({"t": round(t, 4), "R": round(r, 6)})
        
    df_rk = pd.DataFrame(filas_rk)
    
    # CORRECTO: El tiempo X es el valor de 't' (segundos) donde la función alcanzó su máximo
    tiempo_x = t 
    
    return tiempo_x, df_rk

# Ejecutamos el cálculo continuo automáticamente
tiempo_x_segundos, tabla_rk = calcular_runge_kutta(paso_h, r_inicial)

# ==========================================
# PANTALLA PRINCIPAL
# ==========================================
st.subheader("🧮 1. Resultados del Modelo Continuo")
st.info(f"**El tiempo de vencimiento de las vacunas de gripe (X) es:** `{round(tiempo_x_segundos, 2)}` **segundos** ({round(tiempo_x_segundos/60, 2)} minutos).")

with st.expander("📄 Ver Tabla Completa de Runge-Kutta"):
    st.dataframe(tabla_rk, use_container_width=True)

st.markdown("---")
st.subheader("🚀 2. Motor de Simulación Discreta")
if st.button("▶️ Iniciar Simulación", type="primary"):
    with st.spinner("Procesando la simulación... ⏳"):
        import random
        
        # --- 1. FUNCIONES GENERADORAS ---
        def generar_llegada(media):
            rnd = random.random()
            tiempo = -media * np.log(1 - rnd)
            return rnd, tiempo

        def generar_grupo():
            rnd = random.random()
            cantidad = 1 + int(rnd * 4) 
            return rnd, cantidad
            
        # --- 2. INICIALIZACIÓN (Fila 0) ---
        reloj = 0.0
        iteracion = 0
        filas_mostrar = []
        
        estado_enfermero = "Libre"
        cola_covid = 0
        cola_gripe = 0
        caja_gripe_abierta = False
        dosis_gripe = 0
        turno_covid = True 
        
        # --- VARIABLES ESTADÍSTICAS FINALES ---
        tot_vacunados_covid = 0
        tot_vacunados_gripe = 0
        tot_dosis_vencidas = 0
        cajas_covid_abiertas = 0
        cajas_gripe_abiertas = 0
        
        ac_tiempo_ocupado = 0.0
        ac_tiempo_limpieza = 0.0
        ac_area_cola_covid = 0.0
        ac_area_cola_gripe = 0.0
        reloj_anterior = 0.0

        # Primeros Eventos 
        rnd_lleg_cov, t_lleg_cov = generar_llegada(media_llegada_covid)
        prox_llegada_covid = reloj + t_lleg_cov
        
        rnd_lleg_gri, t_lleg_gri = generar_llegada(media_llegada_gripe)
        prox_llegada_gripe = reloj + t_lleg_gri
        
        prox_fin_vacunacion = float('inf')
        prox_vencimiento_gripe = float('inf')
        prox_inicio_limpieza = frecuencia_limpieza
        prox_fin_limpieza = float('inf')

        # Fila 0
        if fila_j == 0:
            filas_mostrar.append({
                "Iter.": iteracion, "Reloj": round(reloj, 4), "Evento": "Inicialización",
                "Llegaron": "-", "Próx Lleg COVID": round(prox_llegada_covid, 4),
                "Próx Lleg Gripe": round(prox_llegada_gripe, 4),
                "Próx Fin Vacunación": "-", "Próx Venc Gripe": "-",
                "Próx Limpieza": round(prox_inicio_limpieza, 4),
                "Cola COVID": cola_covid, "Cola Gripe": cola_gripe,
                "Enfermero": estado_enfermero,
                "Estado Caja Gripe": "Cerrada", "Dosis Gripe Disp": "-",
                "Ac. Vacunados COVID": 0, "Ac. Vacunados Gripe": 0,
                "Ac. Dosis Vencidas": 0, "Cajas COVID Abiertas": 0
            })

        # --- 3. MOTOR DE EVENTOS (BUCLE WHILE) ---
        while iteracion < iteraciones_max:
            iteracion += 1
            
            tiempos = {
                "Llegada COVID": prox_llegada_covid, "Llegada Gripe": prox_llegada_gripe,
                "Fin Vacunacion": prox_fin_vacunacion, "Vencimiento Gripe": prox_vencimiento_gripe,
                "Inicio Limpieza": prox_inicio_limpieza, "Fin Limpieza": prox_fin_limpieza
            }
            evento_actual = min(tiempos, key=tiempos.get)
            reloj = tiempos[evento_actual]

            # --- CÁLCULO DE ESTADÍSTICAS SÍNCRONAS (Áreas y Tiempos UTN) ---
            delta_t = reloj - reloj_anterior
            ac_area_cola_covid += cola_covid * delta_t
            ac_area_cola_gripe += cola_gripe * delta_t
            
            if estado_enfermero in ["Vacunando COVID", "Vacunando Gripe"]:
                ac_tiempo_ocupado += delta_t
            elif estado_enfermero == "Limpiando":
                ac_tiempo_limpieza += delta_t
                
            reloj_anterior = reloj

            rnd_lleg_cov_actual, rnd_lleg_gri_actual = "-", "-"
            llegaron_pacientes = "-"
            
            # B. Procesar Eventos
            if evento_actual == "Llegada COVID":
                rnd_lleg_cov_actual, t_lleg_cov = generar_llegada(media_llegada_covid)
                rnd_cant, cant = generar_grupo()
                llegaron_pacientes = cant
                cola_covid += cant
                prox_llegada_covid = reloj + t_lleg_cov

            elif evento_actual == "Llegada Gripe":
                rnd_lleg_gri_actual, t_lleg_gri = generar_llegada(media_llegada_gripe)
                rnd_cant, cant = generar_grupo()
                llegaron_pacientes = cant
                cola_gripe += cant
                prox_llegada_gripe = reloj + t_lleg_gri

            elif evento_actual == "Fin Vacunacion":
                estado_enfermero = "Libre"
                prox_fin_vacunacion = float('inf')

            elif evento_actual == "Vencimiento Gripe":
                tot_dosis_vencidas += dosis_gripe 
                caja_gripe_abierta = False
                dosis_gripe = 0
                prox_vencimiento_gripe = float('inf')

            elif evento_actual == "Inicio Limpieza":
                estado_enfermero = "Limpiando"
                # Solo se interrumpe la atención humana, NO el tiempo continuo
                if prox_fin_vacunacion != float('inf'):
                    prox_fin_vacunacion += duracion_limpieza
                
                # ELIMINAMOS LAS DOS LÍNEAS QUE DESPLAZABAN EL VENCIMIENTO DE LA GRIPE
                
                prox_inicio_limpieza = reloj + frecuencia_limpieza
                prox_fin_limpieza = reloj + duracion_limpieza

            elif evento_actual == "Fin Limpieza":
                estado_enfermero = "Libre"
                prox_fin_limpieza = float('inf')

            # C. Lógica del Enfermero
            if estado_enfermero == "Libre":
                if turno_covid or cola_gripe == 0:
                    if cola_covid >= 5:
                        cajas_covid_abiertas += 1
                        tot_vacunados_covid += 5
                        
                        estado_enfermero = "Vacunando COVID"
                        cola_covid -= 5
                        prox_fin_vacunacion = reloj + (tiempo_vacunacion_seg * 5) / 60.0 
                        turno_covid = False 
                    elif cola_gripe > 0:
                        turno_covid = False 

                if not turno_covid and estado_enfermero == "Libre":
                    if cola_gripe > 0:
                        if not caja_gripe_abierta:
                            cajas_gripe_abiertas += 1
                            caja_gripe_abierta = True
                            dosis_gripe = 10
                            prox_vencimiento_gripe = reloj + (tiempo_x_segundos / 60.0) 

                        a_vacunar = min(cola_gripe, dosis_gripe)
                        tot_vacunados_gripe += a_vacunar 
                        
                        estado_enfermero = "Vacunando Gripe"
                        cola_gripe -= a_vacunar
                        dosis_gripe -= a_vacunar
                        prox_fin_vacunacion = reloj + (tiempo_vacunacion_seg * a_vacunar) / 60.0

                        if dosis_gripe == 0:
                            caja_gripe_abierta = False
                            prox_vencimiento_gripe = float('inf')
                        
                        turno_covid = True
                    elif cola_covid >= 5:
                        turno_covid = True

            # D. Filtro Visual de Filas
            if (fila_j <= iteracion < (fila_j + filas_i)) or (iteracion == iteraciones_max):
                filas_mostrar.append({
                    "Iter.": iteracion, 
                    "Reloj": round(reloj, 4), 
                    "Evento": evento_actual,
                    "Llegaron": llegaron_pacientes, 
                    "Próx Lleg COVID": round(prox_llegada_covid, 4),
                    "Próx Lleg Gripe": round(prox_llegada_gripe, 4),
                    "Próx Fin Vacunación": round(prox_fin_vacunacion, 4) if prox_fin_vacunacion != float('inf') else "-",
                    "Próx Venc Gripe": round(prox_vencimiento_gripe, 4) if prox_vencimiento_gripe != float('inf') else "-",
                    "Próx Limpieza": round(prox_inicio_limpieza, 4),
                    "Cola COVID": cola_covid, 
                    "Cola Gripe": cola_gripe,
                    "Enfermero": estado_enfermero,
                    "Estado Caja Gripe": "Abierta" if caja_gripe_abierta else "Cerrada",
                    "Dosis Gripe Disp": dosis_gripe if caja_gripe_abierta else "-",
                    "Ac. Vacunados COVID": tot_vacunados_covid,
                    "Ac. Vacunados Gripe": tot_vacunados_gripe,
                    "Ac. Dosis Vencidas": tot_dosis_vencidas,
                    "Cajas COVID Abiertas": cajas_covid_abiertas
                })

        # --- 4. RENDERIZAR TABLA FINAL ---
        st.success(f"✅ ¡Simulación de {iteraciones_max} eventos completada con éxito!")
        df_simulacion = pd.DataFrame(filas_mostrar)
        st.dataframe(df_simulacion.style.format(precision=4), use_container_width=True)

        # --- 5. RENDERIZAR ESTADÍSTICAS OBLIGATORIAS (FASE 5) ---
        st.markdown("---")
        st.subheader("📊 3. Estadísticas Finales (Las 8 obligatorias)")
        
        # Fórmulas de la cátedra para promedios y porcentajes
        porcentaje_ocupacion = (ac_tiempo_ocupado / reloj) * 100 if reloj > 0 else 0
        promedio_cola_covid = ac_area_cola_covid / reloj if reloj > 0 else 0
        promedio_cola_gripe = ac_area_cola_gripe / reloj if reloj > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Vacunados COVID", tot_vacunados_covid)
        col2.metric("2. Vacunados Gripe", tot_vacunados_gripe)
        col3.metric("3. Dosis Gripe Vencidas", tot_dosis_vencidas)
        col4.metric("4. Cajas COVID Abiertas", cajas_covid_abiertas)
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("5. Cajas Gripe Abiertas", cajas_gripe_abiertas)
        col6.metric("6. % Ocupación Enfermero", f"{round(porcentaje_ocupacion, 2)} %")
        col7.metric("7. Promedio Cola COVID", round(promedio_cola_covid, 2))
        col8.metric("8. Promedio Cola Gripe", round(promedio_cola_gripe, 2))