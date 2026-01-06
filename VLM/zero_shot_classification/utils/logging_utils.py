"""
Utilitaires pour le logging
"""

import logging
import sys
from datetime import datetime


def setup_logger(log_file=None, level=logging.INFO):
    """
    Configure le logger pour l'application
    
    Args:
        log_file: Chemin du fichier de log (optionnel)
        level: Niveau de logging
        
    Returns:
        Logger configuré
    """
    # Créer le logger
    logger = logging.getLogger('VLM_ZeroShot')
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
