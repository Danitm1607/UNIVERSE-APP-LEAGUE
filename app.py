import pickle
import pandas as pd
import streamlit as st

# Cargar el dataframe ajustado
try:
    data = pd.read_csv('dataframe_con_predicciones_ajustado.csv')
    st.write("Dataframe cargado correctamente")
except FileNotFoundError:
    st.error("No se encontró el archivo dataframe_con_predicciones_ajustado.csv")
except Exception as e:
    st.error(f"Error al cargar dataframe_con_predicciones_ajustado.csv: {e}")

# Remover nombres de las características
data_values = data.values

# Cargar los modelos
modelo_fast2 = None
modelo_pos = None

try:
    with open('modelo_fast2.pkl', 'rb') as file:
        modelo_fast2 = pickle.load(file)
    st.write("Modelo fast2 cargado correctamente")
except FileNotFoundError:
    st.error("No se encontró el archivo modelo_fast2.pkl")
except Exception as e:
    st.error(f"Error al cargar modelo_fast2.pkl: {e}")

try:
    with open('modelo_pos.pkl', 'rb') as file:
        modelo_pos = pickle.load(file)
    st.write("Modelo pos cargado correctamente")
except FileNotFoundError:
    st.error("No se encontró el archivo modelo_pos.pkl")
except Exception as e:
    st.error(f"Error al cargar modelo_pos.pkl: {e}")

# Función para hacer predicciones
def hacer_prediccion(modelo, datos):
    if modelo is not None:
        try:
            return modelo.predict(datos)
        except AttributeError as e:
            st.error(f"El objeto no es un modelo de sklearn: {e}")
        except Exception as e:
            st.error(f"Error al hacer predicción: {e}")
    else:
        return "Modelo no cargado correctamente"

# Interfaz de Streamlit
st.title("Predicciones de Machine Learning")
st.write("Aquí puedes ingresar datos para obtener predicciones.")

# Verificar si los modelos son cargados correctamente
if modelo_fast2 is None or modelo_pos is None:
    st.error("Uno o ambos modelos no se cargaron correctamente. Por favor verifica los archivos.")

# Ejemplo de uso de la función de predicción
if modelo_fast2 is not None:
    try:
        resultado_fast2 = hacer_prediccion(modelo_fast2, data_values)
        st.write("Resultados del modelo rápido:", resultado_fast2)
    except Exception as e:
        st.error(f"Error al hacer predicción con modelo_fast2: {e}")

if modelo_pos is not None:
    try:
        resultado_pos = hacer_prediccion(modelo_pos, data_values)
        st.write("Resultados del modelo posicional:", resultado_pos)
    except Exception as e:
        st.error(f"Error al hacer predicción con modelo_pos: {e}")
