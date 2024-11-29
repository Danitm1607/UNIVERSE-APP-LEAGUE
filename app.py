import pickle
import pandas as pd
import streamlit as st

# Función para cargar los modelos
def cargar_modelo(modelo_path):
    try:
        with open(modelo_path, 'rb') as file:
            modelo = pickle.load(file)
        st.write(f"Modelo {modelo_path} cargado correctamente")
        return modelo
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {modelo_path}")
    except Exception as e:
        st.error(f"Error al cargar {modelo_path}: {e}")
    return None

# Cargar los modelos
modelo_fast2 = cargar_modelo('modelo_fast2.pkl')
modelo_pos = cargar_modelo('modelo_pos.pkl')

# Función para hacer predicciones
def hacer_prediccion(modelo, datos):
    if modelo is not None:
        try:
            return modelo.predict(datos)
        except Exception as e:
            st.error(f"Error al hacer predicción: {e}")
    else:
        return "Modelo no cargado correctamente"

# Crear una lista de pistas disponibles
pistas = [
    'Abu Dhabi', 'Australia', 'Austria', 'Bahrein', 'Baku', 'Belgica',
    'Brasil', 'Canada', 'China', 'España', 'Estados Unidos', 'Francia',
    'Holanda', 'Inglaterra', 'Japón', 'Jeddah', 'Las Vegas', 'Mexico',
    'Miami', 'Monaco', 'Monza', 'Portugal', 'Qatar', 'Singapur'
]

# Crear una lista de pilotos disponibles
pilotos = [
    'ERICK 195352', 'TXSHURAXX', 'ELPAPURRI540', 'BMR-ADRIAN', 'MXMARIANO',
    'JAMES ORTIZ9763', 'NICOMAGALDI1985', 'SKY-SEVERAL', 'STEBAN_FBE', 'DANTERO22',
    # Agrega los demás pilotos aquí...
]

# Interfaz de Streamlit
st.title("Predicciones de F1")

# Formulario para ingresar nombre del piloto y seleccionar pista
with st.form(key='prediccion_form'):
    piloto = st.selectbox("Selecciona el Piloto", pilotos)
    pista = st.selectbox("Selecciona la Pista", pistas)
    submit_button = st.form_submit_button(label='Buscar')

# Verificar si se ha enviado el formulario
if submit_button:
    if piloto and pista:
        # Aquí debes cargar el dataframe con las características necesarias
        # Asumiendo que 'dataframe_con_predicciones_ajustado_codificado' es el dataframe correcto
        datos = dataframe_con_predicciones_ajustado_codificado[
            (dataframe_con_predicciones_ajustado_codificado['piloto'] == piloto) &
            (dataframe_con_predicciones_ajustado_codificado['pista'] == pista)
        ]
        
        if not datos.empty:
            # Hacer predicciones
            resultado_fast2 = hacer_prediccion(modelo_fast2, datos.values)
            resultado_pos = hacer_prediccion(modelo_pos, datos.values)

            # Mostrar resultados
            st.write(f"Pronóstico de posición para {piloto} en {pista}:", resultado_pos)
            st.write(f"Pronóstico de tiempo de vuelta para {piloto} en {pista}:", resultado_fast2)
        else:
            st.error(f"No se encontraron datos para {piloto} en {pista}.")
    else:
        st.error("Por favor, ingresa el nombre del piloto y selecciona una pista.")
