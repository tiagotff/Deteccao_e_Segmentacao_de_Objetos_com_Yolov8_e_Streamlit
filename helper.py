from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

import settings


def load_model(model_path):
    """
    Carrega um modelo de detecção de objeto YOLO do model_path especificado.

    Parâmetros:
        model_path (str): O caminho para o arquivo de modelo YOLO.

    Retorna:
        Um modelo de detecção de objetos YOLO.
    """

    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Exibe o tracker", ('Sim', 'Não'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Exiba os objetos detectados em um vídeo usando o modelo YOLOv8.

    Argumentos:
    - conf (float): Limite de confiança para detecção de objetos.
    - modelo (YoloV8): Um modelo de detecção de objetos YOLOv8.
    - st_frame (objeto Streamlit): Um objeto Streamlit para exibir o vídeo detectado.
    - imagem (matriz numpy): Uma matriz numpy que representa o quadro do vídeo.
    - is_display_tracking (bool): Um sinalizador que indica se o rastreamento de objetos deve ser exibido (padrão=Nenhum).

    Retorna:
    Nenhum
    """

    # Redimensione a imagem para um tamanho padrão
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Exibe o rastreamento de objetos, se especificado
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Preveja os objetos na imagem usando o modelo YOLOv8
        res = model.predict(image, conf=conf)

    ## Plote os objetos detectados no vídeo
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Vídeo detectado',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Reproduz um stream de webcam. Detecta objetos em tempo real usando o modelo de detecção de objetos YOLOv8.

    Parâmetros:
        conf: Confiança do modelo YOLOv8.
        modelo: uma instância da classe `YOLOv8` contendo o modelo YOLOv8.

    Retorna:
        Nenhum
    """
    source_youtube = st.sidebar.text_input("URL Vídeo YouTube")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detectar Objetos'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Reproduz um stream rtsp. Detecta objetos em tempo real usando o modelo de detecção de objetos YOLOv8.

    Parâmetros:
        conf: Confiança do modelo YOLOv8.
        modelo: uma instância da classe `YOLOv8` contendo o modelo YOLOv8.

    Retorna:
        Nenhum
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('URL de exemplo: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detectar Objetos'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Erro ao carregar o fluxo RTSP: " + str(e))


def play_webcam(conf, model):
   """
    Reproduz um stream de webcam. Detecta objetos em tempo real usando o modelo de detecção de objetos YOLOv8.

    Parâmetros:
        conf: Confiança do modelo YOLOv8.
        modelo: uma instância da classe `YOLOv8` contendo o modelo YOLOv8.

    Retorna:
        Nenhum
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detectar Objetos'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Erro carregando video: " + str(e))


def play_stored_video(conf, model):
    """
    Reproduz um arquivo de vídeo armazenado. Rastreia e detecta objetos em tempo real usando o modelo de detecção de objetos YOLOv8.

    Parâmetros:
        conf: Confiança do modelo YOLOv8.
        modelo: uma instância da classe `YOLOv8` contendo o modelo YOLOv8.

    Retorna:
        Nenhum
    """
    source_vid = st.sidebar.selectbox(
        "Escolha um video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detectar Objetos do Vídeo'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Erro carregando video: " + str(e))
