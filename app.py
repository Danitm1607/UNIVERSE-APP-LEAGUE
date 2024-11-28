import pickle
import pandas as pd
import streamlit as st

# Cargar el dataframe
data = pd.read_csv('dataframe_con_predicciones.csv')

# Cargar los modelos
with open('modelo_fast2.pkl', 'rb') as file:
    modelo_fast2 = pickle.load(file)
with open('modelo_pos.pkl', 'rb') as file:
    modelo_pos = pickle.load(file)

# Función para hacer predicciones
def hacer_prediccion(modelo, datos):
    return modelo.predict(datos)

# Interfaz de Streamlit
st.title("Predicciones de Machine Learning")
st.write("Aquí puedes ingresar datos para obtener predicciones.")

# Ejemplo de uso de la función de predicción
resultado_fast2 = hacer_prediccion(modelo_fast2, data)
st.write("Resultados del modelo rápido:", resultado_fast2)

resultado_pos = hacer_prediccion(modelo_pos, data)
st.write("Resultados del modelo posicional:", resultado_pos)
