from pathlib import Path
import sys

# Obtém o caminho absoluto do arquivo atual
file_path = Path(__file__).resolve()

# Obtém o diretório pai do arquivo atual
root_path = file_path.parent

# Adicione o caminho raiz à lista sys.path se ainda não estiver lá
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Obtém o caminho relativo do diretório raiz em relação ao diretório de trabalho atual
ROOT = root_path.relative_to(Path.cwd())

# Fontes
IMAGE = 'Imagem'
VIDEO = 'Vídeo'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
}

# Configuração do Modelo ML
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
# No caso do seu modelo personalizado comente a linha acima e
# Coloque o nome do arquivo pt do seu modelo personalizado na linha abaixo
# DETECTION_MODEL = MODEL_DIR / 'meu_modelo_detecção.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
