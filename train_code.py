import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import tqdm
import numpy as np
import logging
import sys
from utils import compute_top_k,compute_captioning_metrics
import copy





FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)




def train( epochs,model, train_loader, test_loader,optimizer, scheduler,criterion,model_name,device,captioning,use_vlm):
    
    print("Training Start")
    phases = ['train','test']
    
    best_acc = 0.0
    for epoch in range(epochs):
        data_loaders = {'train':train_loader,'test':test_loader}
        det_losses = {phase: 0.0 for phase in phases}
        det_labels_targets = []
        det_labels_pred = []
        #phases = ['test','test']
        for phase in phases:
            training = phase == 'train'
            model.train(training)
            data_loader = data_loaders[phase]
            with torch.set_grad_enabled(training):
                pbar = tqdm.tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))


                for i, data in enumerate(pbar,start=1):
                    
                    
                    
                    
                    frames,labels,text = data
                    frames = frames.float().to(device)
                    #frames = {k: v.float().to(device) for k, v in frames.items()}
                    labels = labels.long().to(device)
                    
                    
                    





                    
                    if not use_vlm:
                        outputs,_ = model(frames)
                    else:
                        if captioning:
                            outputs,decoded_texts = model(frames,text)
                        else:
                            outputs = model(frames,text)
                    
                    
                    if captioning:
                        det_loss = outputs.loss
                    else:
                        B,C = outputs.size()
                        
                        
                        det_loss = criterion(outputs, labels) 
                    det_losses[phase] += (det_loss.item())# * B)
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })
                    if training:
                        optimizer.zero_grad()
                        det_loss.backward()
                        optimizer.step()
                        scheduler.step()
                    else:


                        if captioning:
                            print(decoded_texts,'pred')
                            print(text,'in')
                            labels_target = text#.detach().cpu()#.numpy()
                            det_labels_targets.extend(labels_target)

                            labels_pred = decoded_texts#.detach().cpu()#.numpy()#.tolist()
                            det_labels_pred.extend(labels_pred)


                        else:
                            
                            labels_target = labels.detach().cpu().numpy()#.tolist()
                            det_labels_targets.extend(labels_target)


                            
                            labels_pred = outputs.detach().cpu().numpy()#.tolist()
                            det_labels_pred.extend(labels_pred)
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in phases:
            if captioning:
                
                capt_metric = compute_captioning_metrics(det_labels_pred,det_labels_targets)
                bleu,rouge,meteor,em = capt_metric['BLEU'],capt_metric['ROUGE-L'],capt_metric['METEOR'],capt_metric['Exact Match']


                if bleu > best_acc :
                    best_acc = bleu
                    # Assuming 'model' is wrapped in DataParallel
                    if isinstance(model, torch.nn.DataParallel):
                        model_save = model.module  # Access the underlying model (without DataParallel wrapper)
                    else:
                        model_save = model
                    model_save = copy.deepcopy(model_save)
                    # Move the model to CPU
                    model_save.to('cpu')

                    # Save the model's state_dict
                    torch.save(model_save.state_dict(), 'new_'+model_name+'.pth')




                log.append('test det_loss: {:.5f} det_bleu: {:.5f} det_rouge: {:.5f} det_meteor: {:.5f} det_em: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset),
                bleu,rouge,meteor,em,
            ))
            else:

                acc = compute_top_k(det_labels_pred,det_labels_targets,1)
                if acc > best_acc :
                    best_acc = acc
                    # Assuming 'model' is wrapped in DataParallel
                    if isinstance(model, torch.nn.DataParallel):
                        model_save = model.module  # Access the underlying model (without DataParallel wrapper)
                    else:
                        model_save = model
                    model_save = copy.deepcopy(model_save)
                    # Move the model to CPU
                    model_save.to('cpu')

                    # Save the model's state_dict
                    torch.save(model_save.state_dict(), 'new_'+model_name+'.pth')
                

                
                
                log.append('test det_loss: {:.5f} det_mAP1: {:.5f}'.format(
                    det_losses['test'] / len(data_loaders['test'].dataset),
                    acc,
                ))
        logger.info(' | '.join(log))
        