import os
from math import ceil
import logging
from pathlib import Path
import json
import time
from time import gmtime, strftime
import importlib.util
import sys

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch import nn

sys.path.append("./ViLReF/")
from clip import load
from clip.model import convert_weights, convert_state_dict, resize_pos_embed, CLIP
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.feat_extract_img import feat_extract_img


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return True  # Always True when not using distributed


def debug(rank):
    args = parse_args()

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Set output path
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    args.log_path = os.path.join(args.logs, args.name, "out_{}.log".format(time_suffix))

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    assert args.precision in ['amp', 'fp16', 'fp32']

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level, 0)  # rank is 0 in non-distributed
    setup_worker_logging(0, log_queue, args.log_level)

    # Build the model
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v
    model_info['use_flash_attention'] = args.use_flash_attention

    model = CLIP(**model_info)
    if args.clip_weight_path is not None:
        assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
    if args.bert_weight_path is not None:
        assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
    load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path,
         use_flash_attention=args.use_flash_attention)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)

    model = model.to(device)
    if args.precision == "fp16":
        convert_weights(model)

    if args.grad_checkpointing:
        assert not torch_version_str_compare_lessequal(torch.__version__, "1.8.0"), \
            "Currently our grad_checkpointing is not compatible with torch version <= 1.8.0."
        model.set_grad_checkpointing()
        logging.info("Grad-checkpointing activated.")

    if args.use_flash_attention:
        assert importlib.util.find_spec("flash_attn"), "flash_attn is not installed."
        logging.info("Using FlashAttention.")

    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.freeze_vision:
        for k, v in model.visual.named_parameters():
            v.requires_grad = False
        # freeze bn running mean and variance
        if args.vision_model in ['RN50']:
            for m in model.visual.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        logging.info("The visual encoder is freezed during training.")

    # Automatic Mixed Precision or FP16
    if args.precision == "fp16":
        convert_weights(model)

    # Automatically restore latest checkpoint if exists
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(f"=> begin to load checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu")
            sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
            # Resize the positional embedding by interpolation, if needed
            resize_pos_embed(sd, model, prefix="module.")
            # Adapt flash attention
            if args.use_flash_attention:
                sd = convert_state_dict(sd)
            # Load the state dict
            model.load_state_dict(sd, False)
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    target_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    if args.use_visual:
        del target_model.bert
    elif args.use_bert:
        del target_model.visual


    extractor = feat_extract_img(model)

    # Example image processing
    imgs = torch.randn([1, 3, 224, 224]).to(device)
    feat = extractor(imgs)


def main():
    debug(0)  # No need for multiprocess spawning


if __name__ == "__main__":
    main()
