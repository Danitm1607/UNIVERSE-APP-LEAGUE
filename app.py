import pickle
import pandas as pd
import streamlit as st

# Cargar el dataframe
try:
    data = pd.read_csv('dataframe_con_predicciones.csv')
except FileNotFoundError:
    st.error("No se encontró el archivo dataframe_con_predicciones.csv")
except Exception as e:
    st.error(f"Error al cargar dataframe_con_predicciones.csv: {e}")

# Cargar los modelos
modelo_fast2 = None
modelo_pos = None

try:
    with open('modelo_fast2.pkl', 'rb') as file:
        modelo_fast2 = pickle.load(file)
except FileNotFoundError:
    st.error("No se encontró el archivo modelo_fast2.pkl")
except Exception as e:
    st.error(f"Error al cargar modelo_fast2.pkl: {e}")

try:
    with open('modelo_pos.pkl', 'rb') as file:
        modelo_pos = pickle.load(file)
except FileNotFoundError:
    st.error("No se encontró el archivo modelo_pos.pkl")
except Exception as e:
    st.error(f"Error al cargar modelo_pos.pkl: {e}")

# Función para hacer predicciones
def hacer_prediccion(modelo, datos):
    if modelo is not None:
        return modelo.predict(datos)
    else:
        return "Modelo no cargado correctamente"

# Interfaz de Streamlit
st.title("Predicciones de Machine Learning")
st.write("Aquí puedes ingresar datos para obtener predicciones.")

# Ejemplo de uso de la función de predicción
try:
    resultado_fast2 = hacer_prediccion(modelo_fast2, data)
    st.write("Resultados del modelo rápido:", resultado_fast2)
except Exception as e:
    st.error(f"Error al hacer predicción con modelo_fast2: {e}")

try:
    resultado_pos = hacer_prediccion(modelo_pos, data)
    st.write("Resultados del modelo posicional:", resultado_pos)
except Exception as e:
    st.error(f"Error al hacer predicción con modelo_pos: {e}")
