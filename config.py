import os

from dotenv import load_dotenv


def init_env():
    load_dotenv(override=True)

def get_config():
    return {
        'MAX_MESSAGES_DISTANCE':            float(os.getenv('MAX_MESSAGES_DISTANCE', '0.3')),
        'BUTTERFLY_MODEL_TIMESTAMPS':       int(os.getenv('BUTTERFLY_MODEL_TIMESTAMPS', '10')),
        'BUTTERFLY_MODEL_WEIGHTS_FILENAME': os.getenv('BUTTERFLY_MODEL_WEIGHTS_FILENAME', 'butterfly_model_weights.pt'),
        'CAT_MODEL_TIMESTAMPS':             int(os.getenv('CAT_MODEL_TIMESTAMPS', '25')),
        'CAT_MODEL_WEIGHTS_FILENAME':       os.getenv('CAT_MODEL_WEIGHTS_FILENAME', 'cat_model_weights.pt'),
        'DEVICE':                           os.getenv('DEVICE', 'cuda'),
        'MODELS_BASE_DIR':                  os.getenv('MODELS_BASE_DIR', './weights'),
        'RABBITMQ_HOST':                    os.getenv("RABBITMQ_HOST", "localhost"),
        'RABBITMQ_PORT':                    int(os.getenv("RABBITMQ_PORT", "5672")),
        'RABBITMQ_USER':                    os.getenv("RABBITMQ_USER", "guest"),
        'RABBITMQ_PASSWORD':                os.getenv("RABBITMQ_PASSWORD", "guest"),
        'RABBITMQ_VIRTUAL_HOST':            os.getenv("RABBITMQ_VIRTUAL_HOST", "/"),
    }

