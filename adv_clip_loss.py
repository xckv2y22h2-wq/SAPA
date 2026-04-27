import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast

from modified_clip import clip
from models.model import *
from models.prompters import TokenPrompter, NullPrompter, PromptLearner
from attacks import *

import torch.nn.functional as F
import numpy as np
import torch.nn as nn



def FT_TeCoA_loss(images, target, text_tokens, optimizer, model, original_model,
                  prompter, add_prompter, prompt_learner, args):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.add_prompt_size == 0: # visual prompt token -- token level
        prompt_token = None    
    else:
        prompt_token = add_prompter()
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    ### Adv generation ###
    delta = attack_pgd(prompter, model, add_prompter, criterion, images,
                       target, text_tokens, alpha, attack_iters, 'l_inf', epsilon=args.train_eps)
    adv_img = clip_img_preprocessing(images + delta)
    prompted_adv_images = prompter(adv_img)
    ### Adv generation ###

    # Classification loss
    output_Iadv_Tnat, _ = multiGPU_CLIP(model, prompted_adv_images, text_tokens, prompt_token)
    loss_cls = criterion(output_Iadv_Tnat, target)

    loss = loss_cls
    return loss, output_Iadv_Tnat


def FT_PMG_loss(images, target, text_tokens, optimizer, model, original_model,
                prompter, add_prompter, prompt_learner, args):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.add_prompt_size == 0: # visual prompt token -- token level
        prompt_token = None    
    else:
        prompt_token = add_prompter()
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    ### Adv generation ###
    delta = attack_pgd(prompter, model, add_prompter, criterion, images,
                       target, text_tokens, alpha, attack_iters, 'l_inf', epsilon=args.train_eps)
    adv_img = clip_img_preprocessing(images + delta)
    prompted_adv_images = prompter(adv_img)
    ### Adv generation ###

    ### Clean sample and its prediction ###
    nat_img = clip_img_preprocessing(images)
    prompted_nat_images = prompter(nat_img)
    with torch.no_grad():
        Ori_output_Inat_Tnat, _ = multiGPU_CLIP(original_model, prompted_nat_images, text_tokens, prompt_token)
    ### Clean sample and its prediction ###

    # Classification loss
    output_Iadv_Tnat, _ = multiGPU_CLIP(model, prompted_adv_images, text_tokens, prompt_token)
    loss_cls = criterion(output_Iadv_Tnat, target)

    # Pred Alignment to the original model loss
    criterion_KL = nn.KLDivLoss(reduction='batchmean').cuda()
    loss_Pred_Align_Ori = criterion_KL(F.log_softmax(output_Iadv_Tnat, dim=1),
                                       F.softmax(Ori_output_Inat_Tnat, dim=1))
    
    # Pred Alignment (for the current model) loss
    output_Inat_Tnat, _ = multiGPU_CLIP(model, prompted_nat_images, text_tokens, prompt_token)
    loss_Pred_Align = criterion_KL(F.log_softmax(output_Iadv_Tnat, dim=1),
                                   F.softmax(output_Inat_Tnat, dim=1))

    loss = loss_cls + args.W_Pred_Align * loss_Pred_Align + args.W_Pred_Align_Ori * loss_Pred_Align_Ori
    return loss, output_Iadv_Tnat


def FT_ImgText_PGD_loss(images, target, text_tokens, optimizer, model, original_model,
                        prompter, add_prompter, prompt_learner, args):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.add_prompt_size == 0: # visual prompt token -- token level
        prompt_token = None    
    else:
        prompt_token = add_prompter()
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    ### Clean sample and its prediction ###
    nat_img = clip_img_preprocessing(images)
    prompted_nat_images = prompter(nat_img)
    with torch.no_grad():
        Ori_output_Inat_Tnat, _ = multiGPU_CLIP(original_model, prompted_nat_images, text_tokens, prompt_token)
    ### Clean sample and its prediction ###

    ### Adv (Img & Text) generation ###
    prompt_learner.load_state_dict(args.original_prompter_state)
    delta = attack_pgd_adv_prompt(prompter, model, add_prompter, criterion, images,
                                  target, text_tokens, alpha, attack_iters, 'l_inf', 
                                  prompt_learner, args.text_perb_stepsize, epsilon=args.train_eps)
    adv_img = clip_img_preprocessing(images + delta)
    prompted_adv_images = prompter(adv_img)
    ### Adv (Img & Text) generation ###

    output_Iadv_Tnat, _ = multiGPU_CLIP_Text_Prompt_Tuning(model, prompted_adv_images, text_tokens, prompt_token, prompt_learner)

    # Classification loss
    loss_cls = criterion(output_Iadv_Tnat, target)

    # Pred Alignment to the original model loss
    criterion_KL = nn.KLDivLoss(reduction='batchmean').cuda()
    loss_Pred_Align_Ori = criterion_KL(F.log_softmax(output_Iadv_Tnat, dim=1),
                                       F.softmax(Ori_output_Inat_Tnat, dim=1))
    
    loss = loss_cls + args.W_Pred_Align_Ori * loss_Pred_Align_Ori
    return loss, output_Iadv_Tnat
    


def FT_TRADES_loss(images, target, text_tokens, optimizer, model, original_model,
                   prompter, add_prompter, prompt_learner, args):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.add_prompt_size == 0: # visual prompt token -- token level
        prompt_token = None    
    else:
        prompt_token = add_prompter()
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    ### Clean sample and its prediction ###
    nat_img = clip_img_preprocessing(images)
    prompted_nat_images = prompter(nat_img)
    with torch.no_grad():
        Ori_output_Inat_Tnat, _ = multiGPU_CLIP(original_model, prompted_nat_images, text_tokens, prompt_token)
    ### Clean sample and its prediction ###

    ### Adv generation ###
    delta = attack_TRADES_KL(prompter, model, add_prompter, criterion, images,
                             target, text_tokens, alpha, attack_iters, 'l_inf', 
                             Ori_output_Inat_Tnat, epsilon=args.train_eps)
    adv_img = clip_img_preprocessing(images + delta)
    prompted_adv_images = prompter(adv_img)
    ### Adv generation ###

    # Multiplicative noise for image and text embeddings
    if args.mul_noise_beta > 0.0:
        output_Iadv_Tnat, _ = multiGPU_CLIP_multiply_noise(model, prompted_adv_images, text_tokens, prompt_token, beta=args.mul_noise_beta)
    else:
        output_Iadv_Tnat, _ = multiGPU_CLIP(model, prompted_adv_images, text_tokens, prompt_token)

    # Nat classification loss
    output_Inat_Tnat, _ = multiGPU_CLIP(model, prompted_nat_images, text_tokens, prompt_token)
    loss_nat_cls = criterion(output_Inat_Tnat, target)

    # Pred Alignment to the original model loss
    criterion_KL = nn.KLDivLoss(reduction='batchmean').cuda()
    loss_Pred_Align_Ori = criterion_KL(F.log_softmax(output_Iadv_Tnat, dim=1), 
                                       F.softmax(Ori_output_Inat_Tnat, dim=1))
    
    loss = loss_nat_cls + args.W_Pred_Align_Ori * loss_Pred_Align_Ori
    return loss, output_Iadv_Tnat


def criterion_L2(out, targets, reduction='mean'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    return squared_error_batch

def FT_FARE_loss(images, target, text_tokens, optimizer, model, original_model,
                   prompter, add_prompter, prompt_learner, args):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.add_prompt_size == 0: # visual prompt token -- token level
        prompt_token = None    
    else:
        prompt_token = add_prompter()
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    ### Clean sample and its prediction ###
    nat_img = clip_img_preprocessing(images)
    prompted_nat_images = prompter(nat_img)
    with torch.no_grad():
        Ori_output_Inat_Tnat, _, Ori_emb_Inat_Tnat, _ = multiGPU_CLIP(original_model, prompted_nat_images, text_tokens, prompt_token, is_embedding=True)
    ### Clean sample and its prediction ###

    ### Adv generation ###
    delta = attack_FARE_Emb_L2(prompter, model, add_prompter, criterion, images,
                             target, text_tokens, alpha, attack_iters, 'l_inf', 
                             Ori_emb_Inat_Tnat, epsilon=args.train_eps)
    adv_img = clip_img_preprocessing(images + delta)
    prompted_adv_images = prompter(adv_img)
    ### Adv generation ###

    # Multiplicative noise for image and text embeddings
    if args.mul_noise_beta > 0.0:
        output_Iadv_Tnat, _ = multiGPU_CLIP_multiply_noise(model, prompted_adv_images, text_tokens, prompt_token, beta=args.mul_noise_beta)
    else:
        output_Iadv_Tnat, _, emb_Iadv_Tnat, _ = multiGPU_CLIP(model, prompted_adv_images, text_tokens, prompt_token, is_embedding=True)

    loss = criterion_L2(emb_Iadv_Tnat, Ori_emb_Inat_Tnat)

    return loss, output_Iadv_Tnat

