import os
import sys
import logging
from logging_config import setup_logging
import transformers
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np

setup_logging()

logger = logging.getLogger(__name__)

logger.info("CLIP model")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")










