from pathlib import Path
import io
import sys
import os

import math
import numpy as np

import requests

import json

import kornia.augmentation as K
from base64 import b64encode
from omegaconf import OmegaConf
import imageio
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from taming.models import cond_transformer, vqgan
import transformers

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

try:
    import clip
except ImportError:
    from CLIP import clip

from utils import *
from upscale_dream import ScaleImage

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
        self.settings = {
            # required
            'seed': -1,
            'prompt': '',
            'width': 256,
            'height': 256,
            'clip_model': 'ViT-B/32', # available ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32','ViT-B/16']
            'vqgan_model': 'vqgan_imagenet_f16_16384', # available ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "sflckr"]
            'initial_image': '',
            'target_images': '',
            'input_images': '',
            'output_folder': 'vqgan-steps',
            'output_name': '',
            'noise_prompt_seeds': [],
            'noise_prompt_weights': [],

            'key_frames': True,
            'generate_video': False,
            'upscale_dream': False,
            'upscale_strength': 2,
            'video_length': 60, # seconds
            'target_fps': 30,
            'iterations_per_frame': 3,
            'angle': 0,
            'zoom': 1,
            'translation_x': 0,
            'translation_y': 0,
            'display_frequency': 10,

            # additional
            'vq_init_weight': 0.0,
            'vq_step_size': 0.1,
            'vq_cutn': 64,
            'vq_cut_pow': 1.0,

            # model links
            'pretrained_models': {
                'vqgan_imagenet_f16_1024_ckpt': 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1',
                'vqgan_imagenet_f16_1024_yaml': 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1',
                'vqgan_imagenet_f16_16384_ckpt': 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1',
                'vqgan_imagenet_f16_16384_yaml': 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1',
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
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{self.dir_path}/{self.settings['output_folder']}/", exist_ok=True)

        self.down_pretrained_models()

        self.replace_grad = ReplaceGrad.apply
        self.clamp_with_grad = ClampWithGrad.apply

        self.model_name = self.settings['vqgan_model']
        self.total_iterations = self.settings['video_length'] * self.settings['target_fps']

        self.clean_cache()

        if self.settings['seed'] == -1:
            self.seed = None
        else:
            self.seed = self.settings['seed']

        if self.settings['key_frames']:
            try:
                self.prompts = self.settings['prompt']
                self.prompts_series = split_key_frame_text_prompts(
                    parse_key_frames(self.settings['prompt']),
                    self.total_iterations
                )
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `prompt` correctly for key frames.\n"
                )
                self.prompts = f"0: ({self.settings['prompt']}:1)"
                self.prompts_series = split_key_frame_text_prompts(
                    parse_key_frames(self.settings['prompt']),
                    self.total_iterations
                )

            try:
                self.target_images = self.settings['target_images']
                self.target_images_series = split_key_frame_text_prompts(
                    parse_key_frames(self.settings['target_images']),
                    self.total_iterations
                )
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `target_images` correctly for key frames.\n"
                )
                self.target_images = f"0: ({self.settings['target_images']}:1)"
                self.target_images_series = split_key_frame_text_prompts(
                    parse_key_frames(self.settings['target_images']),
                    self.total_iterations
                )

            try:
                self.angle_series = get_inbetweens(parse_key_frames(self.settings['angle']), self.total_iterations)
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `angle` correctly for key frames.\n"
                )
                self.angle = f"0: ({self.settings['angle']})"
                self.angle_series = get_inbetweens(parse_key_frames(self.settings['angle']), self.total_iterations)

            try:
                self.zoom_series = get_inbetweens(parse_key_frames(self.settings['zoom']), self.total_iterations)
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `zoom` correctly for key frames.\n"
                )
                self.zoom = f"0: ({self.settings['zoom']})"
                self.zoom_series = get_inbetweens(parse_key_frames(self.settings['zoom']), self.total_iterations)

            try:
                self.translation_x_series = get_inbetweens(parse_key_frames(self.settings['translation_x']), self.total_iterations)
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_x` correctly for key frames.\n"
                )
                self.translation_x = f"0: ({self.settings['translation_x']})"
                self.translation_x_series = get_inbetweens(parse_key_frames(self.settings['translation_x']), self.total_iterations)

            try:
                self.translation_y_series = get_inbetweens(parse_key_frames(self.settings['translation_y']), self.total_iterations)
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_y` correctly for key frames.\n"
                )
                self.translation_y = f"0: ({self.settings['translation_y']})"
                self.translation_y_series = get_inbetweens(parse_key_frames(self.settings['translation_y']), self.total_iterations)
            try:
                self.iterations_per_frame_series = get_inbetweens(
                    parse_key_frames(self.settings['iterations_per_frame']), self.total_iterations, integer=True
                )
            except RuntimeError:
                self.display_message(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `iterations_per_frame` correctly for key frames.\n"
                )
                self.iterations_per_frame = f"0: ({self.settings['iterations_per_frame']})"
                
                self.iterations_per_frame_series = get_inbetweens(
                    parse_key_frames(self.settings['iterations_per_frame']), self.total_iterations, integer=True
                )
        else:
            self.prompts = [phrase.strip() for phrase in self.settings['prompt'].split("|")]
            if self.prompts == ['']:
                self.prompts = []
            self.target_images = self.settings['target_images']
            if self.target_images == "None" or not self.target_images:
                self.target_images = []
            else:
                self.target_images = self.target_images.split("|")
                self.target_images = [image.strip() for image in self.target_images]

            self.angle = float(self.settings['angle'])
            self.zoom = float(self.settings['zoom'])
            self.translation_x = float(self.settings['translation_x'])
            self.translation_y = float(self.settings['translation_y'])
            self.iterations_per_frame = int(self.settings['iterations_per_frame'])

        self.clean_cache()
        for var in ['device', 'model', 'perceptor', 'z']:
            try:
                del globals()[var]
            except:
                pass

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

        self.model = self.load_vqgan_model(self.vqgan_config, self.vqgan_checkpoint)
        if torch.cuda.device_count() > 1:
            self.display_message(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model, device_ids=[_id for _id in range(torch.cuda.device_count())])
            self.model.to(self.device)
            self.model = self.model.module
        else:
            self.model.to(self.device)
        self.perceptor = clip.load(self.settings['clip_model'], jit=False)[0].eval().requires_grad_(False).to(self.device)

        self.cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        self.f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(self.cut_size, self.settings['vq_cutn'], cut_pow=self.settings['vq_cut_pow'])
        self.n_toks = self.model.quantize.n_e
        self.toksX, self.toksY = self.settings['width'] // self.f, self.settings['height'] // self.f
        self.sideX, self.sideY = self.toksX * self.f, self.toksY * self.f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        self.next_loop_stop = False # ensure proper stop for GPU mem

        for i in range(self.total_iterations):
            if self.next_loop_stop:
                break

            if self.settings['key_frames']:
                self.prompts = self.prompts_series[i]
                self.prompts = [phrase.strip() for phrase in self.prompts.split("|")]
                if self.prompts == ['']:
                    self.prompts = []
                self.settings['prompt'] = self.prompts

                self.target_images = self.target_images_series[i]
                if self.target_images == "None" or not self.target_images:
                    self.target_images = []
                else:
                    self.target_images = self.target_images.split("|")
                    self.target_images = [image.strip() for image in self.target_images]
                self.settings['target_images'] = self.target_images

                self.angle = self.angle_series[i]
                self.zoom = self.zoom_series[i]
                self.translation_x = self.translation_x_series[i]
                self.translation_y = self.translation_y_series[i]
                self.iterations_per_frame = self.iterations_per_frame_series[i]

            if i == 0 and self.settings['initial_image'] != "":
                self.img_0 = read_image_workaround(self.settings['initial_image'])
                self.z, *_ = self.model.encode(TF.to_tensor(self.img_0).to(self.device).unsqueeze(0) * 2 - 1)
            elif i == 0 and not os.path.isfile(f'{self.dir_path}/{self.settings["output_folder"]}/{i:04}.png'):
                self.one_hot = F.one_hot(
                    torch.randint(self.n_toks, [self.toksY * self.toksX], device=self.device), self.n_toks
                ).float()
                self.z = self.one_hot @ self.model.quantize.embedding.weight
                self.z = self.z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(0, 3, 1, 2)
            else:
                self.img_0 = read_image_workaround(f'{self.dir_path}/{self.settings["output_folder"]}/{i:04}.png')

                self.center = (1 * self.img_0.shape[1]//2, 1 * self.img_0.shape[0]//2)
                self.trans_mat = np.float32(
                    [[1, 0, self.translation_x],
                    [0, 1, self.translation_y]]
                )
                self.rot_mat = cv2.getRotationMatrix2D(self.center, self.angle, self.zoom)

                self.trans_mat = np.vstack([self.trans_mat, [0,0,1]])
                self.rot_mat = np.vstack([self.rot_mat, [0,0,1]])
                self.transformation_matrix = np.matmul(self.rot_mat, self.trans_mat)

                self.img_0 = cv2.warpPerspective(
                    self.img_0,
                    self.transformation_matrix,
                    (self.img_0.shape[1], self.img_0.shape[0]),
                    borderMode=cv2.BORDER_WRAP
                )
                self.z, *_ = self.model.encode(TF.to_tensor(self.img_0).to(self.device).unsqueeze(0) * 2 - 1)
            
            i += 1

            self.z_orig = self.z.clone()
            self.z.requires_grad_(True)
            self.opt = optim.Adam([self.z], lr=self.settings['vq_step_size'])

            self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                            std=[0.26862954, 0.26130258, 0.27577711])

            self.pMs = []

            for prompt in self.prompts:
                txt, weight, stop = parse_prompt(prompt)
                self.embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
                self.pMs.append(Prompt(self.embed, self.replace_grad, weight, stop).to(self.device))

            for prompt in self.target_images:
                path, weight, stop = parse_prompt(prompt)
                self.img = resize_image(Image.open(path).convert('RGB'), (self.sideX, self.sideY))
                self.batch = self.make_cutouts(TF.to_tensor(self.img).unsqueeze(0).to(self.device))
                self.embed = self.perceptor.encode_image(self.normalize(self.batch)).float()
                self.pMs.append(Prompt(self.embed, self.replace_grad, weight, stop).to(self.device))

            for seed, weight in zip(self.settings['noise_prompt_seeds'], self.settings['noise_prompt_weights']):
                gen = torch.Generator().manual_seed(seed)
                self.embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
                self.pMs.append(Prompt(self.embed, self.replace_grad, weight).to(self.device))

            try:
                self.dream(i)
            except KeyboardInterrupt:
                self.next_loop_stop = True
                pass

        # upscale/gen video
        if self.settings['upscale_dream']:
            scale_settings = {
                'input': f'{self.dir_path}/{self.settings["output_folder"]}',
                'output': f'{self.dir_path}/{self.settings["output_folder"]}-upscaled',
                'suffix': ''
            }
            out_folder = f'{self.dir_path}/{self.settings["output_folder"]}-upscaled'
            ScaleImage(scale_settings)
        else:
            out_folder = f'{self.dir_path}/{self.settings["output_folder"]}'
        if self.settings['generate_video']:
            frames_to_video(out_folder, f'{self.dir_path}/out.mp4', self.settings['target_fps'])

    def dream(self, i):
        x = 0
        while True:
            if x >= self.iterations_per_frame:
                break
            else:
                self.train(i)
            x += 1

    def train(self, i):
        self.opt.zero_grad()
        lossAll = self.ascend_txt(i, True)
        if i % self.settings['display_frequency'] == 0:
            self.checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))
    
    def save_output(self, i, img):
        if self.settings['output_name'] == '':
            filename = f"{self.dir_path}/{self.settings['output_folder']}/{i:04}.png"
        else:
            filename = f"{self.dir_path}/{self.settings['output_folder']}/{self.settings['output_name']}.png"
        imageio.imwrite(filename, np.array(img))

    def ascend_txt(self, i, save):
        out = self.synth(self.z)
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.settings['vq_init_weight']:
            result.append(F.mse_loss(self.z, self.z_orig) * self.settings['vq_init_weight'] / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))

        if save:
            self.save_output(i, img)
        
        return result

    @torch.no_grad()
    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        self.display_message(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        TF.to_pil_image(out[0].cpu()).save('progress.png')

    def vector_quantize(self, x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        
        return self.replace_grad(x_q, x)

    def synth(self, z):
        z_q = self.vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        
        return self.clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

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
        self.display_message(f'Model down {self.settings["vqgan_model"]} in progress')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        self.display_message(f'Model down {self.settings["vqgan_model"]} done')

    def down_pretrained_models(self):
        models_path = f'{self.dir_path}/models/'
        os.makedirs(models_path, exist_ok=True)

        if os.path.exists(models_path + self.settings['vqgan_model'] + '.ckpt') == False:
            self.stream_down(
                self.settings['pretrained_models'][f'{self.settings["vqgan_model"]}_ckpt'],
                models_path + f'{self.settings["vqgan_model"]}.ckpt'
            )
            self.stream_down(
                self.settings['pretrained_models'][f'{self.settings["vqgan_model"]}_yaml'],
                models_path + f'{self.settings["vqgan_model"]}.yaml'
            )

# total iterations = video_length * target_fps
# key_frames = True, allows setup such as 10: (Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)
# from frame 0 to frame 10 show Apple, from frame 10 to 20 show Orange & Peach
if __name__ == "__main__":
    settings = {
        'key_frames': True,
        'generate_video': True,
        'video_length': 6,
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
