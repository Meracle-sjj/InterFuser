"""
[INPUT]: 依赖本地文本、JSON、图像与 NumPy 样本文件，并接受数据集根路径。
[OUTPUT]: 对外提供 BaseIODataset，统一封装文本/图像/JSON/NumPy 读取与历史帧回退。
[POS]: timm.data 的底层本地 I/O 边界，被 CARLA dataset 复用，不参与划分或标签决策。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

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
        with open(self.root_path + path, "r") as stream:
            return stream.read()

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
            with open(self.root_path + path) as stream:
                json_value = json.load(stream)
        except Exception as e:
            _logger.info(path)
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            with open(self.root_path + new_path) as stream:
                json_value = json.load(stream)
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
