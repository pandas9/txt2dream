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
"""
total iterations = video_length * target_fps
key_frames = True, allows setup such as 10: (Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)
from frame 0 to frame 10 show Apple, from frame 10 to 20 show Orange & Peach
"""

from txt2dream import Text2Image
settings = {
    'key_frames': True,
    'width': 256,
    'height': 256,
    'prompt': '10: (Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)',
    'angle': '10: (0), 30: (10), 50: (0)',
    'zoom': '10: (1), 30: (1.2), 50: (1)',
    'translation_x': '0: (0)',
    'translation_y': '0: (0)',
    'iterations_per_frame': '0: (1)'
    'generate_video': True,
    'video_length': 6, # seconds
    'target_fps': 30,
    'upscale_dream': True,
    'upscale_strength': 2, # available [2, 4] -> 2x or 4x the generated output
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
![gen-vid](examples/out.gif?raw=true)
![gen-art](examples/0900_scaled.png?raw=true)

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
