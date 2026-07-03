import io
import json
import os
import logging
import numpy as np
from PIL import Image
import torch

_logger = logging.getLogger(__name__)



class BaseIODataset(torch.utils.data.Dataset):
    def __init__(self, root=''):
        self.root_path = root

    def _load_text(self, path):
        text = open(self.root_path + path, 'r').read()
        return text

    def _load_image(self, path):
        try:
            img = Image.open(self.root_path + path)
        except Exception as e:
            # Try to load previous frames as fallback
            _logger.warning(f"Image not found: {self.root_path + path}, trying previous frames...")
            frame_num = int(path[-8:-4])
            found = False
            
            # Try previous frames (up to 10 frames back)
            for offset in range(1, 11):
                if frame_num - offset < 0:
                    break
                new_path = path[:-8] + "%04d.jpg" % (frame_num - offset)
                try:
                    img = Image.open(self.root_path + new_path)
                    _logger.warning(f"  -> Successfully loaded frame {frame_num - offset}")
                    found = True
                    break
                except:
                    continue
            
            if not found:
                _logger.error(f"Could not find any valid frame for {path}")
                raise FileNotFoundError(f"No valid frame found for {self.root_path + path}")
        
        return img

    def _load_json(self, path):
        try:
            json_value = json.load(open(self.root_path + path))
        except Exception as e:
            _logger.info(path)
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(self.root_path + new_path))
        return json_value

    def _load_npy(self, path):
        try:
            array = np.load(self.root_path + path, allow_pickle=True)
        except Exception as e:
            _logger.info(path)
            n = path[-8:-4]
            new_path = path[:-8] + "%04d.npy" % (int(n) - 1)
            array = np.load(self.root_path + new_path, allow_pickle=True)
        return array
