import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import get_dataset
from unet import UNet 
from models import get_model
from reconstruct_block import quantize_model, QuantizableLayer, reconstruct_block
from quantizer import WeightQuantizer, ActivationQuantizer
from utils import LinearTempDecay, ActivationHook, evaluate_classifier, evaluate_train
import fire
import logging
from collections import OrderedDict
import os
from tqdm import tqdm
import random
import numpy as np
import time as mytime
import higher

reconstruct_unit = (
            'BasicBlock','Bottleneck', # resnet block
            'ResBottleneckBlock', # regnet block
            'DwsConvBlock', 'ConvBlock', # mobilenetb block
            'InvertedResidual', # mobilenetv2, mnasnet block
            'Linear', 'Conv2d' # default
        )
logging.basicConfig(style='{', format='{asctime} {levelname:8} {name:20} {message}', datefmt='%H:%M:%S', level=logging.INFO)

log = logging.getLogger(__name__)
def train_meta(Transform_model, teacher, data_loader, data_loader_val, val_set, bit_w=4, bit_a=4, iterations=20000, iterations_meta=500, lr_T_alpha=0.01, use_wandb=False, alpha=1, beta=1, gamma=1, thereshold=0.2,setting=None,num_samples=1024):
     # define optimizer for Transform_model and quantized model
    augment = augmentation(setting=setting)
    print("Using augmentation",setting)
    student = quantize_model(
    	teacher, bit_w, bit_a,
    	(nn.Conv2d, nn.Linear, nn.Identity)
    )
    for name, module in student.named_modules():
            if isinstance(module, QuantizableLayer):
                module.enable_act_quant = True
                module.enable_weight_quant = True
    image_init = next(iter(data_loader))[0].cuda()
    _ = student(image_init)
      
    acc_train = 0
    acc_test = 0
    KL_loss = torch.nn.KLDivLoss(reduction='batchmean')

    lr_bit = 0.001      
    lr_w_scale = 0.0001
    lr_a_scale = 0.00004
    annealing_warmup = 0.2
    annealing_range=(20,2)
    temp_decay_meta = LinearTempDecay(
        iterations_meta, rel_start_decay=annealing_warmup,
        start_t=annealing_range[0], end_t=annealing_range[1])

    final_qmodel = quantize_model(teacher, bit_w, bit_a, (nn.Conv2d, nn.Linear, nn.Identity))
    for name, module in final_qmodel.named_modules():
            if isinstance(module, QuantizableLayer):
            #    module.enable_act_quant = True
                module.enable_weight_quant = True
    final_qmodel_modules = OrderedDict(final_qmodel.named_modules())
    teacher_modules = OrderedDict(teacher.named_modules())
    student_modules = OrderedDict(student.named_modules())
    reconstruct_pair = []
   # meta_list = []
    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name)) 
    total_layer = len(reconstruct_pair)

    CE = nn.CrossEntropyLoss()
   
    for index_block, (teacher_block, student_block, name_meta) in enumerate(reconstruct_pair):
        print(f'Recontruct ({index_block}/{len(reconstruct_pair)}): {name_meta}')
      #  if not "layer4.1" in name_meta:
        start_time1 = mytime.time()
        for module in student_block.modules():
            if isinstance(module, (QuantizableLayer)):
                module.trained = True
                module.enable_act_quant = True
                module.enable_weight_quant = True
        iters = 0
        epochs = 0
        param_a_scale = []
        param_w_scale = []
        param_bit = []
        for module in student_block.modules():
            if isinstance(module, WeightQuantizer):
                module.train_mode = True
                param_w_scale.append(module.scale)
                param_bit.append(module.bit_logit)
            elif isinstance(module, ActivationQuantizer):
                module.train_mode = True
                if module.scale is not None:
                    param_a_scale.append(module.scale)

        opt_full = torch.optim.Adam([
            {"params": param_w_scale, 'lr': lr_w_scale},
            {"params": param_a_scale, 'lr': lr_a_scale},
            {"params": param_bit, 'lr': lr_bit}
        ]) 
        

        global include
        module_list, name_list, include = [], [], False
        module_list, name_list = find_unquantized_module(student, module_list, name_list)
        for name, module in student.named_modules():
            if isinstance(module, QuantizableLayer):
                if not module.trained:
                    module.set_quant_state(False,False)
        optimizer_Transform_model = torch.optim.Adam(
            filter(lambda p: p.requires_grad, Transform_model.parameters()),
            lr_T_alpha) 
        scheduler_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_Transform_model, 'min', patience=100, factor=0.95, verbose=True)
        skip_modules = ["conv1", "fc", "classifier.1", "features.0.0"]
        # Use the skip_modules list in the condition
        if not any(skip in name_meta for skip in skip_modules):
            iters = 0
            iter_meta = 0
            while iters < iterations_meta:
                epochs += 1
                loss_save = []
                loss_meta_val_save = []
                for _, (image, labels) in enumerate(data_loader):
                    iters += 1
                    image, labels = image.cuda(), labels.cuda()
                    for module in student_block.modules():
                        if isinstance(module, (WeightQuantizer,ActivationQuantizer)):
                            module.train_mode = True
                    iter_meta +=1
                    if torch.cuda.is_available():
                        student.cuda()
                        Transform_model = Transform_model.cuda()
                    #   student.train()
                    optimizer_Transform_model.zero_grad()
                    with higher.innerloop_ctx(student, opt_full) as (qmetamodel, diffopt):
                        qmetamodel_modules = OrderedDict(qmetamodel.named_modules())
                        
                        qmeta_block = qmetamodel_modules[name_meta]
                        for module in qmeta_block.modules():
                            if isinstance(module, (WeightQuantizer)):
                                module.train_mode = True
                        # forward pass to compute the initial weighted los
                        x_modify = Transform_model(image)
                        
                        t_hook = ActivationHook(teacher_block)
                        f_hook = ActivationHook(qmeta_block)
                        out_meta = qmetamodel(x_modify)
                        with torch.no_grad():
                            out_fp = teacher(x_modify)

                        round_loss = 0
                        annealing_temp_meta = temp_decay_meta(iter_meta)
                        annealing_warmup = 0.2
                        if iter_meta >= annealing_warmup*iterations_meta:
                            for module in qmeta_block.modules():
                                if isinstance(module, WeightQuantizer):
                                    round_loss += (1 - (2*module.soft_target() - 1).abs().pow(annealing_temp_meta)).sum()
                        act_x_meta = t_hook.inputs
                        act_y_meta = t_hook.outputs
                        act_x_q_meta = f_hook.inputs
                        t_hook.remove()
                        f_hook.remove()
                        y_q = qmeta_block(act_x_q_meta)
                        recon_loss_qmetamodel = (y_q - act_y_meta).pow(2).sum(1).mean()
                        loss_meta = recon_loss_qmetamodel + round_loss
                        # update quantied meta model
                        diffopt.step(loss_meta)
                        
                        # evaluate the loss
                        qmetamodel.eval()
                        qmeta_block.eval()
                        loss_save.append(loss_meta.detach().cpu().numpy())
                        t_hook_val = ActivationHook(teacher_block)
                        f_hook_val = ActivationHook(qmeta_block)
                        val_iter = iter(data_loader_val)
                        images_val, labels_val = next(val_iter)
                        images_val = images_val.cuda()
                        labels_val = labels_val.cuda()
                        out_meta_val,_,  fea_val_q= qmetamodel(images_val, feat=True)
                        
                        with torch.no_grad():
                            out_real = teacher(image)
                            out_real_val, _,  fea_val_fp = teacher(images_val, feat=True)
                        act_y_val = t_hook_val.outputs
                        act_y_q_meta_val = f_hook_val.outputs
                        f_hook_val.remove()
                        t_hook_val.remove()
                        x_modify_mixup, labels_modify_mixup = mixup(x_modify, labels)
                        image_mix, labels_image_mixup = mixup(image, labels)
                        with torch.no_grad():
                            out_real_mix, _, fea_real = teacher(image_mix, feat=True)
                        reconstruct_val = (act_y_val - act_y_q_meta_val).pow(2).sum(1).mean()
                        out_modfiy_fp, _, feature_generated = teacher(x_modify_mixup, feat=True)
                        kl_los_val = KL_loss(F.log_softmax(out_meta_val / 4, dim=1), F.softmax(out_real_val / 4, dim=1))
                        PKT_LOSS = pkt_loss(feature_generated,fea_real)       
                        loss_meta_val = alpha*kl_los_val + beta*F.relu(thereshold - (x_modify - image).pow(2).mean()) + gamma*PKT_LOSS 
                        #+ reconstruct_val 
                       
                    grad_of_grads = torch.autograd.grad(loss_meta_val, Transform_model.parameters())
                    # Manually assign gradients to the parameters
                    for param, grad in zip(Transform_model.parameters(), grad_of_grads):
                        param.grad = grad
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(Transform_model.parameters(), 0.1)
                    optimizer_Transform_model.step()
                    scheduler_encoder.step(loss_meta_val)
                    loss_meta_val_save.append(kl_los_val.detach().cpu().numpy())
                    optimizer_Transform_model.zero_grad()

                if use_wandb:
                    import wandb
                    wandb.log({"Val_meta_loss KL": np.mean(loss_meta_val_save)})
                    wandb.log({"Meta_loss": np.mean(loss_save)})
                    print("meta loss:", np.mean(loss_save))
                    print(f"Val meta loss at epoch {epochs} is {np.mean(loss_meta_val_save)}")
                    print(f"weight correlation loss is {PKT_LOSS}")
                else:
                    print("meta loss:", np.mean(loss_save))
                    print(f"Val meta loss at epoch {epochs} is {np.mean(loss_meta_val_save)}")
                    print(f"weight correlation loss is {PKT_LOSS}")
                    

        final_qmodel_blocks = final_qmodel_modules[name_meta]
        for module in final_qmodel_blocks.modules():
            if isinstance(module, (QuantizableLayer)):
                module.trained = True
                module.enable_act_quant = True
                module.enable_weight_quant = True
        t_hook_final = ActivationHook(teacher_block)
        s_hook_final = ActivationHook(final_qmodel_blocks)
        act_x_final, act_y_final, act_x_q_final = [], [], []
        with torch.no_grad():
            for _, (x, labels) in enumerate(data_loader):
                x = x.cuda() # 
                teacher(x)
                final_qmodel(x)
                act_x_final.append(t_hook_final.inputs)
                act_y_final.append(t_hook_final.outputs)
                act_x_q_final.append(s_hook_final.inputs)
                if not any(skip in name_meta for skip in skip_modules):
                    x_new = Transform_model(x)
                    if setting is not None:
                        if setting=="cutmix" or setting=="mixup":
                            x_new, _ = augment(x_new, labels)
                        else:
                            x_new = augment(x_new)
                    teacher(x_new)
                    final_qmodel(x_new)
                    act_x_final.append(t_hook_final.inputs)
                    act_y_final.append(t_hook_final.outputs)
                    act_x_q_final.append(s_hook_final.inputs)
        
        act_x_final = torch.cat(act_x_final)
        act_y_final = torch.cat(act_y_final)
        act_x_q_final = torch.cat(act_x_q_final)
                
        t_hook_final.remove()
        s_hook_final.remove()    
        num_perms = int(num_samples/32)
        reconstruct_block(final_qmodel_blocks, act_x_final, act_x_q_final, act_y_final, batch_size=num_perms, iterations=iterations)
        student_block.load_state_dict(final_qmodel_blocks.state_dict())
        torch.cuda.empty_cache()
    #############################
        for module in student_block.modules():
            if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = False
        for module in final_qmodel_blocks.modules():
            if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = False
        #############################
        print("time for reconstruct block", mytime.time()- start_time1) 
    
    for name, module in student.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = True
            module.enable_weight_quant = True
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False
    evaluate_train(data_loader, student)
    accuracy = evaluate_classifier(val_set, student)
    if use_wandb:
        import wandb
        wandb.log({"val_acc": accuracy})
        
specials_unquantized = [nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout]
def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """Store subsequent unquantized modules in a list"""
    global include
    for name, module in model.named_children():
        if isinstance(module, QuantizableLayer):
            if not module.trained:
                include = True
                module.set_quant_state(False,False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]

import torch.nn.functional as F

    
def augmentation(setting):
    aug = None
    if setting == "contrast":
        print("Using contrast")
        aug =  transforms.ColorJitter(contrast=(0.5, 2.0))
    elif setting == "brightness":
        print("Using brightness")
        aug =  transforms.ColorJitter(brightness=(0.5, 2.0))
    elif setting == "rotation":
        print("Using rotation")
        aug =  transforms.RandomRotation(degrees=30)
    elif setting == "mixup":
        aug =  v2.MixUp(num_classes=1000)
    elif setting == "cutmix":
        aug =  v2.CutMix(num_classes=1000)
    elif setting == "combine":
        aug =  v2.Compose([
            v2.MixUp(num_classes=1000),
            v2.CutMix(num_classes=1000)
        ])
    else:
        print("No augmentation")
    return aug


def pkt_loss(f_s, f_t, eps=1e-7):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(f_s**2, dim=1, keepdim=True))
    f_s = f_s / (output_net_norm + eps)
    f_s[f_s != f_s] = 0
    target_net_norm = torch.sqrt(torch.sum(f_t**2, dim=1, keepdim=True))
    f_t = f_t / (target_net_norm + eps)
    f_t[f_t != f_t] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(f_s, f_s.transpose(0, 1))
    target_similarity = torch.mm(f_t, f_t.transpose(0, 1))
    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(
        model_similarity, dim=1, keepdim=True
    )
    target_similarity = target_similarity / torch.sum(
        target_similarity, dim=1, keepdim=True
    )
    # Calculate the KL-divergence
    loss = torch.mean(
        target_similarity
        * torch.log((target_similarity + eps) / (model_similarity + eps))
    )
    return loss

    
def mixup(val_images, val_labels):

    alpha = 4
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    all_inputs = val_images
    all_targets = F.one_hot(val_labels, num_classes=1000)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = val_images, val_images[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    val_images = mixed_input
    val_labels = mixed_target
    return val_images, val_labels

