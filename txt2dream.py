from pathlib import Path
import io
import sys

import math
import numpy as np

import requests

import json

import kornia.augmentation as K
from base64 import b64encode
from omegaconf import OmegaConf
import imageio
from PIL import ImageFile, Image
from imgtag import ImgTag    # metadata
from libxmp import *         # metadata
import libxmp                # metadata
from stegano import lsb
ImageFile.LOAD_TRUNCATED_IMAGES = True

from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from tqdm.notebook import tqdm

from CLIP import clip

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from utils import *

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape

        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)

        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

class Text2Image:
    '''
    Deep dream text with torch, vqgan, esrgan, clip and diffusion.
    Adjust settings for more illusive dreams.
    '''

    def __init__(self, settings={}):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.down_pretrained_models()

        self.settings = {
            # required
            'seed': -1,
            'display_frequency': 10,
            'use_diffusion': False,
            'prompt': '',
            'width': 256,
            'height': 256,
            'clip_model': 'ViT-B/32', # available ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32','ViT-B/16']
            'vqgan_model': 'vqgan_imagenet_f16_16384', # available ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "sflckr"]
            'initial_image': '',
            'target_images': '',
            'input_images': '',
            'max_iterations': 1000,

            # additional
            'vq_init_weight': 0.0,
            'vq_step_size': 0.1,
            'vq_cutn': 64,
            'vq_cutpow': 1.0,

            # which models to download
            'pretrained_models': {
                'diffusion': False,
                'imagenet_1024': False,
                'imagenet_16384': True,
                'coco': True,
                'wikiart_16384': True,
                'sflckr': True,

                'diffusion_512_ckpt': 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt',
                'imagenet_1024_ckpt': 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1',
                'imagenet_1024_yaml': 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1',
                'imagenet_16384_ckpt': 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1',
                'imagenet_16384_yaml': 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1',
                'coco_ckpt': 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt',
                'coco_yaml': 'https://dl.nmkd.de/ai/clip/coco/coco.yaml',
                'wikiart_16384_ckpt': 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt',
                'wikiart_16384_yaml': 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml',
                'sflckr_ckpt': 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1',
                'sflckr_yaml': 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1',
            }
        }
        for key, value in settings.items():
            self.settings[key] = value

        self.replace_grad = ReplaceGrad.apply
        self.clamp_with_grad = ClampWithGrad.apply

        self.model_name = self.settings['vqgan_model']

        self.clean_cache()

        if self.settings['seed'] == -1:
            self.seed = None

        if self.settings['initial_image'] == 'None' or self.settings['initial_image'] == '':
            self.initial_image = None

        if self.settings['target_images'] == '' or not self.settings['target_images']:
            self.target_images = []
        else:
            self.target_images = self.settings['target_images'].split('|')
            self.target_images = [ image.strip() for image in self.target_images ]

        if self.settings['initial_image'] or self.target_images != []:
            self.settings['input_images'] = True
        
        self.prompts = [ frase.strip() for frase in self.settings['prompt'].split('|') ]
        if self.prompts == ['']:
            self.prompts = []

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.display_message('Deep dream initiated')
        self.display_message(f'Using {self.device}')
        if self.prompts:
            self.display_message(f'I am dreaming about {self.prompts}')
        if self.target_images:
            self.display_message(f'Using dream state {self.target_images}')
        if self.seed == None:
            self.seed = torch.seed()

        torch.manual_seed(self.seed)
        self.display_message(f'Dream seed {self.seed}')

        # config
        self.vqgan_config = f'{self.dir_path}/models/{self.model_name}.yaml'
        self.vqgan_checkpoint = f'{self.dir_path}/models/{self.model_name}.ckpt'
        self.noise_prompt_seeds = []
        self.noise_prompt_weights = []

        self.model = self.load_vqgan_model(self.vqgan_config, self.vqgan_checkpoint).to(self.device)
        self.perceptor = clip.load(self.settings['clip_model'], jit=False)[0].eval().requires_grad_(False).to(self.device)

        self.cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        self.f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(self.cut_size, self.settings['vq_cutn'], cut_pow=self.settings['vq_cutpow'])
        self.n_toks = self.model.quantize.n_e
        self.toksX, self.toksY = self.settings['width'] // self.f, self.settings['height'] // self.f
        self.sideX, self.sideY = self.toksX * self.f, self.toksY * self.f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if self.initial_image:
            self.pil_image = Image.open(self.initial_image).convert('RGB')
            self.pil_image = self.pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
            self.z, *_ = model.encode(TF.to_tensor(self.pil_image).to(self.device).unsqueeze(0) * 2 - 1)
        else:
            self.one_hot = F.one_hot(torch.randint(self.n_toks, [self.toksY * self.toksX], device=self.device), self.n_toks).float()
            self.z = self.one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(0, 3, 1, 2)

        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=self.settings['vq_step_size'])

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        self.pMs = []

        for prompt in self.prompts:
            self.txt, self.weight, self.stop = self.parse_prompt(prompt)
            self.embed = self.perceptor.encode_text(clip.tokenize(self.txt).to(self.device)).float()
            self.pMs.append(Prompt(self.embed, self.replace_grad, self.weight, self.stop).to(self.device))

        for prompt in self.target_images:
            self.path, self.weight, self.stop = parse_prompt(prompt)
            self.img = resize_image(Image.open(self.path).convert('RGB'), (self.sideX, self.sideY))
            self.batch = self.make_cutouts(TF.to_tensor(self.img).unsqueeze(0).to(self.device))
            self.embed = self.perceptor.encode_image(self.normalize(self.batch)).float()
            self.pMs.append(Prompt(self.embed, self.replace_grad, self.weight, self.stop).to(self.device))

        for seed, weight in zip(self.noise_prompt_seeds, self.noise_prompt_weights):
            self.gen = torch.Generator().manual_seed(self.seed)
            self.embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=self.gen)
            self.pMs.append(Prompt(self.embed, self.replace_grad, self.weight).to(self.device))

        self.dream()

    def dream(self):
        i = 0
        try:
            with tqdm() as pbar:
                while True:
                    self.train(i)
                    if i == self.settings['max_iterations']:
                        break
                    i += 1
                    pbar.update()
        except KeyboardInterrupt:
            pass

    def train(self, i):
        self.opt.zero_grad()
        lossAll = self.ascend_txt(i)
        if i % self.settings['display_frequency'] == 0:
            self.checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def ascend_txt(self, i):
        out = self.synth(self.z)
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.settings['vq_init_weight']:
            result.append(F.mse_loss(self.z, self.z_orig) * self.settings['vq_init_weight'] / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        filename = f"{self.dir_path}/vqgan-steps/{i:04}.png"
        imageio.imwrite(filename, np.array(img))
        add_xmp_data(filename, self.prompts, self.model_name, self.seed, self.settings['input_images'], i)
        
        return result

    @torch.no_grad()
    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        self.display_message(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        TF.to_pil_image(out[0].cpu()).save('progress.png')
        add_xmp_data('progress.png', self.prompts, self.model_name, self.seed, self.settings['input_images'], i)

    def vector_quantize(self, x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook

        return self.replace_grad(x_q, x)

    def synth(self, z):
        z_q = self.vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        
        return self.clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    def parse_prompt(self, prompt):
        vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]

        return vals[0], float(vals[1]), float(vals[2])

    def display_message(self, msg):
        print(msg)

    def clean_cache(self):
        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def load_vqgan_model(self, config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del model.loss

        return model

    def stream_down(self, url, path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    def down_pretrained_models(self):
        models_path = f'{self.dir_path}/models/'
        os.makedirs(models_path, exist_ok=False)

        if self.settings['pretrained_models']['diffusion']:
            self.stream_down(
                self.settings['pretrained_models']['diffusion_512_ckpt'],
                models_path
            )

        if self.settings['pretrained_models']['imagenet_1024']:
            self.stream_down(
                self.settings['pretrained_models']['imagenet_1024_ckpt'],
                models_path
            )
            self.stream_down(
                self.settings['pretrained_models']['imagenet_1024_yaml'],
                models_path
            )

        if self.settings['pretrained_models']['imagenet_16384']:
            self.stream_down(
                self.settings['pretrained_models']['imagenet_16384_ckpt'],
                models_path
            )
            self.stream_down(
                self.settings['pretrained_models']['imagenet_16384_yaml'],
                models_path
            )

        if self.settings['pretrained_models']['coco']:
            self.stream_down(
                self.settings['pretrained_models']['coco_ckpt'],
                models_path
            )
            self.stream_down(
                self.settings['pretrained_models']['coco_yaml'],
                models_path
            )

        if self.settings['pretrained_models']['wikiart_16384']:
            self.stream_down(
                self.settings['pretrained_models']['wikiart_16384_ckpt'],
                models_path
            )
            self.stream_down(
                self.settings['pretrained_models']['wikiart_16384_yaml'],
                models_path
            )

        if self.settings['pretrained_models']['sflckr']:
            self.stream_down(
                self.settings['pretrained_models']['sflckr_ckpt'],
                models_path
            )
            self.stream_down(
                self.settings['pretrained_models']['sflckr_yaml'],
                models_path
            )

settings = {
    'prompt': 'World where pyramids fly'
}
Text2Image(settings)
