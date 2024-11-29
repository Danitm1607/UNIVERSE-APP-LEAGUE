import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Cargar los modelos
with open('Modelo_ML_Pos.pkl', 'rb') as file:
    modelo_pos = pickle.load(file)

with open('Modelo_ML_Fast2.pkl', 'rb') as file:
    modelo_fast2 = pickle.load(file)

# Función para hacer predicciones
def hacer_prediccion(modelo, datos):
    return modelo.predict(datos)

# Historial de Predicciones
historial_predicciones = []

# Interfaz de Streamlit
st.title("Predicciones de F1")

# Leer el archivo de datos para obtener las opciones de ID y PISTA
df = pd.read_csv('databaseuniverse (1).csv', delimiter=';')

# Formulario de entrada
with st.form(key='prediccion_form'):
    piloto_id = st.selectbox("Selecciona el ID del Piloto", df['ID'].unique())
    pista = st.selectbox("Selecciona la Pista", df['PISTA'].unique())
    categoria = st.selectbox("Selecciona la Categoría", df['CAT.'].unique())
    submit_button = st.form_submit_button(label='Hacer Predicción')

# Procesar formulario
if submit_button:
    # Filtrar por ID, PISTA y CAT.
    datos_filtrados = df[(df['ID'] == piloto_id) & (df['PISTA'] == pista) & (df['CAT.'] == categoria)]

    # Crear el DataFrame de entrada para el modelo con columnas exactas
    datos = datos_filtrados.copy()
    columnas_necesarias = ['NUMERO REGISTRO', 'ID', 'PILOTO', 'CAT.', 'ESCUDERIA', 'TEMP', 'PAÍS', 'PISTA',
                           'TERM', 'PUN', 'FAST', 'FAST2', 'PROMEDIO PUNTOS', 'MEDIANA PUNTOS',
                           'DESVIACION PUNTOS', 'PROMEDIO POSICION', 'MEDIANA POSICION', 'DESVIACION POSICION',
                           'PROMEDIO VUELTA', 'MEDIANA VUELTA', 'DESVIACION VUELTA', 'DESVIACION VUELTA PILOTO', 
                           'PROMEDIO TIEMPO VUELTA POR PISTA', 'PROM PUNTOS POR TEMPORADA', 'NUM DNF PILOTO', 
                           'PROMEDIO PUNTOS ESCUDERIA POR TEMPORADA', 'PROMEDIO PUNTOS POR ESCUDERIA',
                           'PROMEDIO PUNTOS PILOTO ESCUDERIA', 'NUMERO DE CARRERAS FINALIZADAS', 'PUNTOS POR CARRERA FINALIZADA',
                           'DIF VUELTAS CARRERA', 'PROMEDIO VUELTA PILOTO POR PISTA', 'MEJOR TIEMPO PILOTO PISTA',
                           'PEOR TIEMPO PILOTO PISTA', 'FRECUENCIA DE MEJORA', 'PUNTOS DE EQUIPO', 'INDICE DE COMPETITIVIDAD',
                           'CONSISTENCIA DE PUNTOS', 'PROMEDIO POSICIONES TEMPORADA', 'PROGRESION DE MEJORAS',
                           'RELACION PUNTOS POSICION', 'CARRERAS TOTALES EN UNIVERSE', 'CARRERAS TOTALES EN F1',
                           'CARRERAS TOTALES EN F2', 'CARRERAS TOTALES EN F3', 'CARRERAS EN RED BULL', 'CARRERAS EN FERRARI',
                           'CARRERAS EN MERCEDES', 'CARRERAS EN MCLAREN', 'CARRERAS EN ASTON', 'CARRERAS EN WILLIAMS',
                           'CARRERAS EN ALPINE', 'CARRERAS EN ALPHA ROMEO', 'CARRERAS EN ALPHA TAURI', 'CARRERAS EN HAAS',
                           'PUNTOS TOTALES EN UNIVERSE', 'PUNTOS TOTALES EN F1', 'PUNTOS TOTALES EN F2', 'PUNTOS TOTALES EN F3',
                           'PUNTOS EN TEMPORADA 1', 'PUNTOS EN TEMPORADA 2', 'PUNTOS EN TEMPORADA 3', 'PUNTOS TEMPORADA 4',
                           'PUNTOS TEMPORADA 5', 'PUNTOS TEMPORADA 6', 'PROMEDIO DE PUNTOS F1', 'PROMEDIO DE PUNTOS F2',
                           'PROMEDIO DE PUNTOS F3', 'VICTORIAS TOTALES UNIVERSE', 'VICTORIAS TOTALES F1',
                           'VICTORIAS TOTALES F2', 'VICTORIAS TOTALES F3', 'PODIOS TOTALES UNIVERSE', 'PODIOS TOTALES UNIVERSE F1',
                           'PODIOS TOTALES UNIVERSE F2', 'PODIOS TOTALES UNIVERSE F3']
    
    # Asegurarse de que el DataFrame tiene todas las columnas necesarias
    datos = datos[columnas_necesarias]

    # Convertir todas las columnas a numéricas
    datos = datos.apply(pd.to_numeric, errors='coerce')

    resultado_pos = hacer_prediccion(modelo_pos, datos)
    resultado_fast2 = hacer_prediccion(modelo_fast2, datos)

    # Guardar en el historial de predicciones
    prediccion = {
        'ID': piloto_id,
        'Pista': pista,
        'Categoría': categoria,
        'Predicción de Posición': resultado_pos[0],
        'Predicción de Tiempo de Vuelta': resultado_fast2[0]
    }
    historial_predicciones.append(prediccion)

    # Mostrar los resultados
    st.write(f"Predicción de posición para ID {piloto_id} en {pista}: {resultado_pos[0]}")
    st.write(f"Predicción de tiempo de vuelta para ID {piloto_id} en {pista}: {resultado_fast2[0]} segundos")

# Mostrar el historial de predicciones
st.write("Historial de Predicciones")
st.dataframe(historial_predicciones)

