from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random

import os

from txt2dream import Text2Image

class Image2Animation:
    def __init__(self, settings={}):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))

        self.settings = {
            'frames': 100, # each frame = one image
            'from_frame': 1, # start generating from frame
            'angle': random.randint(12,15),
            'zoom': 15,
            'max_iterations': 60,
            'prompt': '',
            'width': 256,
            'height': 256,
            'initial_image': '', # last generated image
            'output_folder': 'vqgan-animated-steps',
            'crop_coord_left': 3,
            'crop_coord_upper': 3
        }
        for key, value in settings.items():
            self.settings[key] = value
        os.makedirs(f"{self.dir_path}/{self.settings['output_folder']}/", exist_ok=True)
        self.settings['zoom_width'] = self.settings['width'] + self.settings['zoom']
        self.settings['zoom_height'] = self.settings['height'] + self.settings['zoom']
        self.settings['crop_box'] = (self.settings['crop_coord_left'], self.settings['crop_coord_upper'], self.settings['width'], self.settings['height'])

        for i in range(self.settings['from_frame'], self.settings['frames']):
            filename = f'{i:04}'
            _settings = {
                **self.settings,
                'output_name': filename
            }
            if i > 1:
                _settings['initial_image'] = f"{self.dir_path}/{self.settings['output_folder']}/{i -1:04}.png"
            Text2Image(_settings)

            img = Image.open(f"{self.dir_path}/{self.settings['output_folder']}/{filename}.png")
            img_edited = img.rotate(self.settings['angle']).resize((self.settings['zoom_width'], self.settings['zoom_height']), resample=Image.NEAREST).crop(self.settings['crop_box'])
            img_edited.save(f"{self.dir_path}/{self.settings['output_folder']}/{filename}.png")

if __name__ == "__main__":
    settings = {
        'prompt': 'World with flying pyramids',
        'width': 512,
        'height': 512,
        'angle': 0,
    }
    Image2Animation()
