
import torchvision
import loss_utils as lu
import torch
import os



def print_gpu_memory():
    os.system("nvidia-smi | grep -o .*% | grep -oP '\d+MiB /.*MiB.*' > gpu_status.txt")
    with open('gpu_status.txt','r') as f:
        output = f.read()
    print(output, end='')

def metrics_batch(pred, target):
    pred= torch.sigmoid(pred)
    _, metric=dice_loss(pred, target)
    
    return metric

def loss_epoch(model,loss_func,dataset_dl,device,sanity_check=False,memory_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        if memory_check:
            print('Before copy minibatch: ', end = '')
            print_gpu_memory()
        xb=xb.to(device)
        yb=yb.to(device)
        if memory_check:
            print('After copy minibatch: ', end = '')
            print_gpu_memory()

        output=model(xb)#.squeeze()
        loss_b, metric_b=lu.loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric+=metric_b

        if sanity_check is True:
            break
        if memory_check:
            print("After end minibatch: ", end = '')
            print_gpu_memory()
    
    loss=running_loss/float(len_data)
    
    metric=running_metric/float(len_data)
    
    return loss, metric

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

import copy
def train_val(model, params, device, saveFunction):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    memory_check=params["memory_check"]
    memory_minibach_check=params["memory_minibach_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    load_previous_weights=params["load_previous_weights"]
    
    loss_history={
        "train": [],
        "val": []}
    
    metric_history={
        "train": [],
        "val": []}    
    
    val_loss = float('inf')
    if load_previous_weights:
        path2weights="./models/weights.pt"
        model.load_state_dict(torch.load(path2weights))
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,device,sanity_check)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=val_loss    
    if memory_check:
        print('BEFORE EPOCHS GPU MEMORY: ', end='')
        print_gpu_memory()
        print("-"*20)

    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        if memory_check:
            print("BEFORE EPOCH "+str(epoch)+" :", end='')
            print_gpu_memory()

        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,device,sanity_check,memory_minibach_check,opt)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,device,sanity_check)
       
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)   
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            if saveFunction is not None:
                saveFunction(path2weights)
            print("Copied best model weights!")
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 
            
        print("train loss: %.6f, dice: %.4f" %(train_loss,100*train_metric))
        print("val loss: %.6f, dice: %.4f" %(val_loss,100*val_metric))
        if memory_check:
            print("AFTER EPOCH "+str(epoch)+" :", end='')
            print_gpu_memory()

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history        
