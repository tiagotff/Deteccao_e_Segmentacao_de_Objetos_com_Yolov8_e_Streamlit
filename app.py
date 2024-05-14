# Pacotes embutidos
from pathlib import Path
#import PIL

# Pacotes externos
import streamlit as st

# Módulos Locais
import settings
import helper

#Configurando o layout da página
st.set_page_config(
    page_title="Detecção e rastreamento de objetos em tempo real com YOLOv8 e Streamlit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da página principal
st.title("Detecção e rastreamento de objetos em tempo real com YOLOv8 e Streamlit")

# Sidebar
st.sidebar.header("Configuração do Modelo ML")

# Opções de modelo
model_type = st.sidebar.radio(
    "Selecione a Tarefa", ['Detecção', 'Segmentação'])

confidence = float(st.sidebar.slider(
    "Selecione a Confiança do Modelo", 25, 100, 40)) / 100

# Selecionando detecção ou segmentação
if model_type == 'Detecção':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentação':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Carregar modelo de ML pré-treinado
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Não foi possível carregar o modelo. Verifique o caminho especificado: {model_path}")
    st.error(ex)

st.sidebar.header("Conf. Imagem/Vídeo")
source_radio = st.sidebar.radio(
    "Selecione a opção desejada", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Escolha uma imagem...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Imagem Padrão",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Ocorreu um erro ao abrir a imagem.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Imagem com Detecção',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detectar Objetos'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Imagem com Detecção',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("Nenhuma imagem foi carregada ainda!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Selecione um tipo de fonte válido!")
