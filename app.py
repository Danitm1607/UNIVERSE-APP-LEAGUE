import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

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
    'ALUCARDADRI', 'FULLLEGEND741', 'DANITM1607', 'METPAPICHULO', 'ITSRAMSES15',
    'SKY WIZBAND', 'LUCACEDE98', 'ADANBGAYMER', 'DIEGOTZE2000', 'SATED-FOOTMARK1',
    'NICOGONJIM', 'MANOLO765', 'RODOOGAMES', 'AKALVLO', 'LPR STREAK', 'EFS RACING',
    'SEBAS33311', 'SKY-FXSTEER', 'VASCOOL_911', 'ISAIFACUNDO', 'DBARCELONA87967',
    'CARLX9779', 'D I E G OPRO892', 'ALVAROG226', 'GERAMEDRA11', 'JPZMIX',
    'EMMAFOREVER_20', 'JONATHANKYADK89', 'JOTAPE_ORTIZ', 'FREDY1178', 'JAREDMUNDO07',
    'SLPZ333', 'MAGNATE 963211', 'YXDAN', 'FLRA_1992', 'NICO_MAGALDI1985', 'LUIS_OHH-02',
    'AZTK CHUCHO6', 'DANII30101993', 'E_TEJERO7', 'HECTORPAS11', 'KRISTIANLUNA1',
    'YAHIR0511', 'AVILA5609', 'CESARLEOF1', 'INF_ALAN', 'ANGEELBTW', 'ALEJOLEAL',
    'JOSHUA1866', 'THE ALEXG', 'BOOMMER5264', 'DANILO2204', 'KYLO_CL', 'METEORO_GN10',
    'SRS ARTURMAN', 'TURCIOSERWIN6', 'GORDOCRISTIAN14', 'XAVISALAZAR182', 'S4NT8MOON',
    'PAULOLIVAS16', 'GIANFOXRIDER', 'XXWARRIORXX8918', 'KILLERA11AN', 'JOSMANU2004',
    'THEYUKIGAMEZ', 'DAJOSO31', 'STRUCKLEONJR', 'WISECOIN', 'XEL_PIKANTEMAXX_',
    'CTRCAPITANCM', 'MLC YOSOYCHRIS', 'SPOKGANG', 'SANTYVELASKEZ', 'BRUNO UBERA',
    'RONALDRMR', 'JOSEPHSANCHEZ22', 'TIIC ASGAR', 'JOSEANTONIOVR-99', 'XARISTOTELESX',
    'VISIBLEXXX', 'HYBRIDBR1', 'JH14ZS', 'SKY JUMP6615', 'JUAN3DCF', 'SAMDEVIL',
    'EMICHROCKYT', 'LVR CRIS', 'PAULRASVZLA', 'CFC_EZEQUIEL_14', 'MONTANAS4460',
    'ROCKSTAR239376', 'TVC ARCANGEL', 'JG EDU4RD0', 'KARMA2090', 'VG AS3S1N0',
    'ISAAAC ARS', 'THEOMYFOR', 'HUGO ARCHILLA', 'BRANDONGAMER885', 'EDU_RAMIREZ13',
    'RINZLER_VRTX', 'ZTR SNAKE', 'ONE', 'KIKEMADRID8', 'DELUXEE-PLAY', 'NAYIBSK',
    'MARCO ZOSAYA', 'LLXE_MRERICK', 'JJOJ1201', 'THEKINGSONRRIKS', 'ALM_XDD',
    'ALANJACK10', 'RAG0MU01', 'JUANKF14', 'WFC105', 'PAPOROCK2021', 'TIIC MUCINO',
    'VANTONIOBART', 'ALUJANC', 'AMENRIV021', 'PAOLO', 'CESARGP-17', 'CRL LALO',
    'IDX KRISTHIAN RIVERA', 'ANTONY FLORES', 'DOREONIX', 'CF1OMARLO8', 'LKR-TINGLING',
    'TRIANA ANDRES', 'NCR-CIENFUEGOS', 'CYPHER', 'B4SAL W4RR10R', 'TIICDRAKO812',
    'JEFEMAESTROF1', 'XLR8R', 'LAMQ-ZARC', 'DTR ATLAS20', 'EMMABYTWT', 'THEONE_1117',
    'MAVERICK1122', 'LEONARDO062799', 'MENA-S98', 'AZAREDCS1604223', 'CALAN10',
    'SANTIAGO CANOSA', 'MICHAEL DIAZGRANADOS', 'GNUREDSOX'
]

# Codificar las variables categóricas
le_pilotos = LabelEncoder()
le_pilotos.fit(pilotos)
le_pistas = LabelEncoder()
le_pistas.fit(pistas)

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
        # Crear el dataframe basado en el nombre del piloto y la pista seleccionada
        datos = pd.DataFrame({
            'NUMERO REGISTRO': [1],  # Reemplaza con los valores reales
            'TEMP': [2021],  # Reemplaza con los valores reales
            'FECHA': ['2021-03-28'],  # Reemplaza con los valores reales
            'POS': [1],  # Reemplaza con los valores reales
            'PUN': [25],  # Reemplaza con los valores reales
            'FAST': [1],  # Reemplaza con los valores reales
            'FAST2': [1],  # Reemplaza con los valores reales
            'PROMEDIO PUNTOS': [10.5],  # Reemplaza con los valores reales
            'MEDIANA PUNTOS': [10],  # Reemplaza con los valores reales
            'DESVIACION PUNTOS': [3.2],  # Reemplaza con los valores reales
            'PROMEDIO POSICION': [4.5],  # Reemplaza con los valores reales
            'MEDIANA POSICION': [4],  # Reemplaza con los valores reales
            'DESVIACION POSICION': [2.1],  # Reemplaza con los valores reales
            'PROMEDIO VUELTA': [1.30],  # Reemplaza con los valores reales
            'MEDIANA VUELTA': [1.29],  # Reemplaza con los valores reales
            'DESVIACION VUELTA': [0.02],  # Reemplaza con los valores reales
            'DESVIACION VUELTA PILOTO': [0.01],  # Reemplaza con los valores reales
            'PROMEDIO TIEMPO VUELTA POR PISTA': [1.31],  # Reemplaza con los valores reales
            'PROM PUNTOS POR TEMPORADA': [100],  # Reemplaza con los valores reales
            'NUM DNF PILOTO': [0]  # Reemplaza con los valores reales
        })

        # Hacer predicciones
        resultado_fast2 = hacer_prediccion(modelo_fast2, datos.values)
        resultado_pos = hacer_prediccion(modelo_pos, datos.values)

        # Agregar predicciones al DataFrame original
        datos['prediccion_pos'] = resultado_pos
        datos['prediccion_fast2'] = resultado_fast2

        # Mostrar resultados
        st.write(f"Pronóstico de posición para {piloto} en {pista}:", resultado_pos)
        st.write(f"Pronóstico de tiempo de vuelta para {piloto} en {pista}:", resultado_fast2)
        
        # Mostrar el DataFrame con las predicciones
        st.dataframe(datos)
    else:
        st.error("Por favor, ingresa el nombre del piloto y selecciona una pista.")
