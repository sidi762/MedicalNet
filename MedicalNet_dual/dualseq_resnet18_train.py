from setting import parse_opts
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
from datasets.custom_dataset import CustomTumorDataset
from torch.utils.tensorboard import SummaryWriter
from EarlyStopping_torch import EarlyStopping

writer = SummaryWriter()

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets, patience):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)

    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        loss_func = loss_func.cuda()

    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            correct = 0
            total = 0
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            out_class = model(volumes)
            if not sets.no_cuda:
                out_class = out_class.cuda()
                label = label.cuda()

            # calculating loss
            loss = loss_func(out_class, label)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            _, predicted = torch.max(out_class.data, 1)
            correct += (predicted == label).float().sum()
            total += label.size(0)
            accuracy = 100 * correct / total
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, accuracy = {:.3f} avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, last_loss, accuracy, avg_batch_time))

            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                                'ecpoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)

        #Validation per epoch
        with torch.no_grad():
            model.train(False)
            correct = 0
            total = 0
            running_val_loss = 0.0
            for batch_id, batch_data in enumerate(validation_loader):
                batch_id_sp = epoch * batches_per_epoch
                val_volumes, val_labels = batch_data

                if not sets.no_cuda:
                    val_volumes = val_volumes.cuda()

                val_out_class = model(val_volumes)
                if not sets.no_cuda:
                    val_out_class = val_out_class.cuda()
                    val_labels = val_labels.cuda()

                _, predicted = torch.max(val_out_class.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).float().sum()
                val_loss = loss_func(val_out_class, val_labels)
                running_val_loss += val_loss


            val_accuracy = 100 * correct / total
            avg_val_loss = running_val_loss / (batch_id + 1)
            log.info('Validation loss {}'.format(avg_val_loss))
            log.info('Validation accuracy {}'.format(val_accuracy))
            writer.add_scalar("Loss/train", last_loss, epoch)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

            early_stopping(val_loss,model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        #End Validation

    writer.flush()
    print('Finished training')
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt'
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28



    # getting model
    torch.manual_seed(sets.manual_seed)
    sets.model = 'resnet'
    sets.model_depth = 18
    sets.resnet_shortcut = 'A'
    model, parameters = generate_model(sets) #3D Resnet 18
    print (model)
    # optimizer
    if (not sets.ci_test) and sets.pretrain_path:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate },
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
        ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    optimizer = torch.optim.Adam(params,
                                 lr=sets.learning_rate,
                                 betas=(0.9,0.999),
                                 eps=1e-08,
                                 weight_decay=1e-3,
                                 amsgrad=False)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
    #training_dataset = BrainS18Dataset(sets.data_root, sets.img_list, sets)
    training_dataset = CustomTumorDataset(sets.data_root, sets)
    validation_dataset = CustomTumorDataset(sets.data_root_val, sets)
    print('Training set has {} instances'.format(len(training_dataset)))
    print('Validation set has {} instances'.format(len(validation_dataset)))
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    #EarlyStopping
    patience = 300

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets, patience=patience)
    writer.close()
