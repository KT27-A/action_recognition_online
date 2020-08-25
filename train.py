import torch 
from utils import *
import time

def train_epoch(epoch, train_dataloader, model, criterion, optimizer, opts,
                    train_logger):

    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    
    
    for i, (inputs, targets) in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)

        inputs = inputs.to(opts.device)
        targets = targets.to(opts.device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                'Lr {lr}'.format(
                    epoch,
                    i + 1,
                    len(train_dataloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies,
                    lr=optimizer.param_groups[-1]['lr']))

    if opts.log == 1:
        train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[-1]['lr']
        })


def val_epoch(epoch, val_dataloader, model, criterion, optimizer, opts,
                    val_logger, scheduler):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    for i, (inputs, targets) in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)
        inputs = inputs.to(opts.device)
        targets = targets.to(opts.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
    
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Val_Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(val_dataloader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies))
    if opts.log == 1:
        val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    accuracy_val = accuracies.avg
    if accuracy_val > list(opts.highest_val.values())[0]:
        old_key = list(opts.highest_val.keys())[0]
        file_path = os.path.join(opts.result_path, old_key)
        if os.path.exists(file_path):
            os.remove(file_path)
        opts.highest_val.pop(old_key)
        opts.highest_val['save_{}_max.pth'.format(epoch)] = accuracy_val

        save_file_path = os.path.join(opts.result_path,
                                    'save_{}_max.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opts.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
    scheduler.step(losses.avg)
    