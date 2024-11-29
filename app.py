import pickle
import pandas as pd
import streamlit as st

# Cargar los modelos
with open('Modelo_ML_Pos.pkl', 'rb') as file:
    modelo_pos = pickle.load(file)

with open('Modelo_ML_Fast2.pkl', 'rb') as file:
    modelo_fast2 = pickle.load(file)

# Función para hacer predicciones
def hacer_prediccion(modelo, datos):
    return modelo.predict(datos)

# Interfaz de Streamlit
st.title("Predicciones de F1")

# Formulario de entrada
with st.form(key='prediccion_form'):
    piloto = st.selectbox("Selecciona el Piloto", ['Dani', 'Fer', 'Vascool', 'Otro'])
    pista = st.selectbox("Selecciona la Pista", ['Qatar', 'Inglaterra', 'España', 'Otro'])
    temp = st.number_input("Temporada", min_value=1, step=1)
    puntos = st.number_input("Puntos", min_value=0, step=1)
    submit_button = st.form_submit_button(label='Hacer Predicción')

# Procesar formulario
if submit_button:
    # Crear el DataFrame de entrada para el modelo
    datos = pd.DataFrame({
        'NUMERO REGISTRO': [1],  # Reemplaza con valores reales
        'ID': [1],  # Reemplaza con valores reales
        'PILOTO': [1],  # Reemplaza con valores reales
        'CAT.': [0],  # Reemplaza con valores reales
        'ESCUDERIA': [1],  # Reemplaza con valores reales
        'TEMP': [temp],
        'PAÍS': [1],  # Reemplaza con valores reales
        'PISTA': [1],  # Reemplaza con valores reales
        'TERM': [1],  # Reemplaza con valores reales
        'PUN': [puntos],
        'FAST': [1.5],  # Reemplaza con valores reales
        'FAST2': [1.5],  # Reemplaza con valores reales
        'PROMEDIO PUNTOS': [10.5],
        'MEDIANA PUNTOS': [10],
        'DESVIACION PUNTOS': [3.2],
        'PROMEDIO POSICION': [4.5],
        'MEDIANA POSICION': [4],
        'DESVIACION POSICION': [2.1],
        'PROMEDIO VUELTA': [1.3],
        'MEDIANA VUELTA': [1.29],
        'DESVIACION VUELTA': [0.02],
        'DESVIACION VUELTA PILOTO': [0.01],
        'PROMEDIO TIEMPO VUELTA POR PISTA': [1.31],
        'PROM PUNTOS POR TEMPORADA': [100],
        'NUM DNF PILOTO': [0],
        'PROMEDIO PUNTOS ESCUDERIA POR TEMPORADA': [10.5],
        'PROMEDIO PUNTOS POR ESCUDERIA': [10.5],
        'PROMEDIO PUNTOS PILOTO ESCUDERIA': [10.5],
        'NUMERO DE CARRERAS FINALIZADAS': [20],
        'PUNTOS POR CARRERA FINALIZADA': [5],
        'DIF VUELTAS CARRERA': [0.01],
        'PROMEDIO VUELTA PILOTO POR PISTA': [1.3],
        'MEJOR TIEMPO PILOTO PISTA': [1.28],
        'PEOR TIEMPO PILOTO PISTA': [1.35],
        'FRECUENCIA DE MEJORA': [0.5],
        'PUNTOS DE EQUIPO': [200],
        'INDICE DE COMPETITIVIDAD': [0.5],
        'CONSISTENCIA DE PUNTOS': [0.5],
        'PROMEDIO POSICIONES TEMPORADA': [10],
        'PROGRESION DE MEJORAS': [0.5],
        'RELACION PUNTOS POSICION': [0.5],
        'CARRERAS TOTALES EN UNVIERSE': [30],
        'CARRERAS TOTALES EN F1': [20],
        'CARRERAS TOTALES EN F2': [10],
        'CARRERAS TOTALES EN F3': [5],
        'CARRERAS EN RED BULL': [10],
        'CARRERAS EN FERRARI': [5],
        'CARRERAS EN MERCEDES': [7],
        'CARRERAS EN MCLAREN': [8],
        'CARRERAS EN ASTON': [6],
        'CARRERAS EN WILLIAMS': [2],
        'CARRERAS EN ALPINE': [3],
        'CARRERAS EN ALPHA ROMEO': [1],
        'CARRERAS EN ALPHA TAURI': [4],
        'CARRERAS EN HAAS': [0],
        'PUNTOS TOTALES EN UNIVERSE': [300],
        'PUNTOS TOTALES EN F1': [200],
        'PUNTOS TOTALES EN F2': [80],
        'PUNTOS TOTALES EN F3': [20],
        'PUNTOS EN TEMPORADA 1': [50],
        'PUNTOS EN TEMPORADA 2': [60],
        'PUNTOS EN TEMPORADA 3': [70],
        'PUNTOS TEMPORADA 4': [80],
        'PUNTOS TEMPORADA 5': [90],
        'PUNTOS TEMPORADA 6': [100],
        'PROMEDIO DE PUNTOS F1': [10.5],
        'PROMEDIO DE PUNTOS F2': [8.5],
        'PROMEDIO DE PUNTOS F3': [5.5],
        'VICTORIAS TOTALES UNIVERSE': [5],
        'VICTORIAS TOTALES F1': [3],
        'VICTORIAS TOTALES F2': [1],
        'VICTORIAS TOTALES F3': [1],
        'PODIOS TOTALES UNIVERSE': [10],
        'PODIOS TOTALES UNIVERSE F1': [7],
        'PODIOS TOTALES UNIVERSE F2': [2],
        'PODIOS TOTALES UNIVERSE F3': [1]
    })

    # Hacer las predicciones
    resultado_pos = hacer_prediccion(modelo_pos, datos)
    resultado_fast2 = hacer_prediccion(modelo_fast2, datos)

    # Mostrar los resultados
    st.write(f"Predicción de posición para {piloto} en {pista}: {resultado_pos[0]}")
    st.write(f"Predicción de tiempo de vuelta para {piloto} en {pista}: {resultado_fast2[0]}")
