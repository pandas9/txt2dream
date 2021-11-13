# txt2dream
machine vivid dreams

# Requirements
`pip install -r requirements.txt --upgrade`  <br />
if system is not recognizing taming transformers import git clone https://github.com/CompVis/taming-transformers into same folder as txt2dream.py

# Usage
##### Text to image
`txt2dream.py / txt2dream.ipynb`
`python txt2dream.py`
```
# total iterations = video_length * target_fps
# key_frames = True, allows setup such as 10: (Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)
# from frame 0 to frame 10 show Apple, from frame 10 to 20 show Orange & Peach
from txt2dream import Text2Image
settings = {
    'key_frames': True,
    'generate_video': True,
    'video_length': 6, # seconds
    'target_fps': 30,
    'upscale_dream': True,
    'upscale_strength': 2, # available [2, 4] -> 2x or 4x the generated output
    'initial_image': '', # start from image
    'target_images': '', # target the shape
    'prompt': '10: (Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)',
    'width': 256,
    'height': 256,
    'angle': '10: (0), 30: (10), 50: (0)',
    'zoom': '10: (1), 30: (1.2), 50: (1)',
    'translation_x': '0: (0)',
    'translation_y': '0: (0)',
    'iterations_per_frame': '0: (1)',
    'vqgan_model': 'vqgan_imagenet_f16_16384', # available ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "sflckr"]
    'clip_model': 'ViT-B/32' # available ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32','ViT-B/16']
}
Text2Image(settings)
```

##### Upscale generated images
`upscale_dream.py`
`python upscale_dream.py`
```
from upscale_dream import ScaleImage

settings = {
    'input': './vqgan-steps', # './vqgan-steps/0001.png',
    'output': './vqgan-steps-upscaled'
}
ScaleImage(settings)
```

# Examples
[pandas9.github.io/gen-art/](https://pandas9.github.io/gen-art/) <br />
![gen-art](https://github.com/pandas9/gen-art/blob/main/public/choice-illusion/0900_scaled.png?raw=true)

# Docker
Coming soon

# References
This kind of work is possible because of cool people such as <br />
https://github.com/crowsonkb <br />
https://github.com/justin-bennington/S2ML-Art-Generator <br />
https://github.com/xinntao/Real-ESRGAN <br />
https://github.com/CompVis/taming-transformers <br />
https://github.com/lucidrains/big-sleep <br />
https://github.com/openai/CLIP <br />
https://github.com/hojonathanho/diffusion
