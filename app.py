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

# Función para convertir segundos a formato MM:SS
def convertir_tiempo(segundos):
    minutos = int(segundos // 60)
    segundos_restantes = segundos % 60
    return f"{minutos}m {segundos_restantes:.2f}s"

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

    # Crear el DataFrame de entrada para el modelo
    datos = datos_filtrados.copy()
    # Añadir cualquier otra lógica necesaria aquí

    resultado_pos = hacer_prediccion(modelo_pos, datos)
    resultado_fast2 = hacer_prediccion(modelo_fast2, datos)
    tiempo_formateado = convertir_tiempo(resultado_fast2[0])

    # Mostrar los resultados
    st.write(f"Predicción de posición para ID {piloto_id} en {pista}: {resultado_pos[0]}")
    st.write(f"Predicción de tiempo de vuelta para ID {piloto_id} en {pista}: {tiempo_formateado}")

    # Opcional: Comparar con otros pilotos
    comparar = st.multiselect("Comparar con otros IDs", df['ID'].unique(), default=[piloto_id])
    if len(comparar) > 1:
        comparacion_datos = df[(df['ID'].isin(comparar)) & (df['PISTA'] == pista) & (df['CAT.'] == categoria)]
        comparacion_pos = hacer_prediccion(modelo_pos, comparacion_datos)
        comparacion_fast2 = hacer_prediccion(modelo_fast2, comparacion_datos)
        
        # Mostrar la comparación
        comparacion_datos['Predicción de Posición'] = comparacion_pos
        comparacion_datos['Predicción de Tiempo de Vuelta'] = [convertir_tiempo(tiempo) for tiempo in comparacion_fast2]
        
        st.write("Comparación de Predicciones:")
        st.dataframe(comparacion_datos[['ID', 'Predicción de Posición', 'Predicción de Tiempo de Vuelta']])
        
        # Gráfico de Comparación
        fig, ax = plt.subplots()
        for id_piloto in comparar:
            datos_piloto = comparacion_datos[comparacion_datos['ID'] == id_piloto]
            ax.plot(datos_piloto['PISTA'], datos_piloto['Predicción de Tiempo de Vuelta'], marker='o', label=id_piloto)
        
        ax.set_title("Comparación de Tiempos de Vuelta")
        ax.set_xlabel("Pista")
        ax.set_ylabel("Tiempo de Vuelta (MM:SS)")
        ax.legend()
        st.pyplot(fig)
