from abc import ABC, abstractmethod

class BaseVideoGenerator(ABC):
    @abstractmethod
    def generate(self, prompt, img_path):
        """所有模型必须实现这个方法，返回视频 URL 或本地路径"""
        pass