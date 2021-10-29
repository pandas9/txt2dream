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
from txt2dream import Text2Image

settings = {
    'prompt': 'World with flying pyramids',
    'width': 512,
    'height': 512,
}
Text2Image(settings)
```

##### Animate generated image
`dream2animation.py`
`python dream2animation.py`
```
from dream2animation import Image2Animation

settings = {
    'prompt': 'World with flying pyramids',
    'width': 512,
    'height': 512,
    'angle': 30,
    'zoom': 30
}
Image2Animation(settings)
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
![gen art](https://pandas9.github.io/gen-art/public/time-keeper/0009_scaled.png)

# Docker
Coming soon

# References
This kind of work is possible because of cool people such as <br />
https://github.com/justin-bennington/S2ML-Art-Generator <br />
https://github.com/xinntao/Real-ESRGAN <br />
https://github.com/CompVis/taming-transformers <br />
https://github.com/lucidrains/big-sleep <br />
https://github.com/openai/CLIP <br />
https://github.com/hojonathanho/diffusion
