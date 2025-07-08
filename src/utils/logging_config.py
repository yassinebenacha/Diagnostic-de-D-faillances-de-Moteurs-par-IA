import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'results')
LOG_FILE = os.path.join(LOG_DIR, 'project.log')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré pour le projet."""
    return logging.getLogger(name) 