import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

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
        # Crear el dataframe con las características adicionales necesarias
        datos = pd.DataFrame({
            'ID': [f'ID_{piloto.upper()}'],
            'PISTA': [f'PISTA_{pista.upper()}']
            # Asegúrate de agregar aquí las características adicionales que necesita tu modelo
        })

        # Aplicar One-Hot Encoding a las columnas de texto
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_columns = onehot_encoder.fit_transform(datos[['ID', 'PISTA']])
        
        # Crear un dataframe con las columnas codificadas
        encoded_df = pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(['ID', 'PISTA']))

        # Concatenar las características adicionales con las columnas codificadas
        datos_encoded = pd.concat([encoded_df], axis=1)

        # Hacer predicciones
        resultado_fast2 = hacer_prediccion(modelo_fast2, datos_encoded)
        resultado_pos = hacer_prediccion(modelo_pos,
