import torch
import promonet
import numpy as np
import huggingface_hub

generator_checkpoint = huggingface_hub.hf_hub_download(
    repo_id   = 'maxrmorrison/promonet',
    filename  = f'generator-00{promonet.STEPS}.pt',
    local_dir = "./"
)

discriminator_checkpoint = huggingface_hub.hf_hub_download(
    repo_id  = 'maxrmorrison/promonet',
    filename = f'discriminator-00{promonet.STEPS}.pt',
    local_dir = "./"
)

print(
    generator_checkpoint,
    discriminator_checkpoint,
    type(generator_checkpoint),
    type(discriminator_checkpoint)
)