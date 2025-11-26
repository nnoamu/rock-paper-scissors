import sys
import os
import argparse
from configparser import ConfigParser
from pathlib import Path

sys.path.append(os.path.abspath(os.curdir))
sys.path.append(os.path.abspath(os.curdir.join(["src"])))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from datasets import SkinColorDataset
from src.preprocessing import SkinColorSegmenterNetwork
import numpy as np
from tqdm import tqdm
import time

device='cuda' if torch.cuda.is_available() else 'cpu'
print('Training device:', device)

def train_loop(model, dataloader, loss_fn, optimizer):
    size=len(dataloader)
    running_loss=0
    correct_guesses=0
    all=0

    model.train()
    idx=0
    for input, labels in dataloader:
        input=input.to(device)
        labels=labels.to(device)
        output=model(input)
        loss=loss_fn(output, labels)
        running_loss+=loss.item()
        correct_guesses+=sum(abs(output-labels)<0.5)
        all+=len(input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        idx=idx+1
        if idx%60==0:
            print(idx, loss)
    
    total_loss=running_loss/size
    total_accuracy=correct_guesses/all
    print('Train loss:', total_loss)
    print('Train accuracy:', total_accuracy)
    return total_loss, total_accuracy

def test_loop(model, dataloader, loss_fn):
    size=len(dataloader)
    running_loss=0
    correct_guesses=0
    all=0

    model.eval()
    with torch.no_grad():
        for input, labels in dataloader:
            input=input.to(device)
            labels=labels.to(device)
            output=model(input)
            loss=loss_fn(output, labels)
            running_loss+=loss.item()
            correct_guesses+=sum(abs(output-labels)<0.5)
            all+=len(input)
    
    total_loss=running_loss/size
    total_accuracy=correct_guesses/all
    print('Validation loss:', total_loss)
    print('Validation accuracy:', total_accuracy)
    return total_loss, total_accuracy

def train(path: str):
    NUM_LAYERS=4
    LAYER_DENSITY=10

    NUM_EPOCHS=5
    BATCH_SIZE=2048
    LEARNING_RATE=0.02
    SCALE_FACTOR=0.1
    PATIENCE=2

    config_file=Path(path).joinpath('config.txt')
    config=ConfigParser()
    writer=SummaryWriter(Path(path).joinpath('log'))

    if len(config.read([config_file]))>0:
        if 'num_layers' in config['NETWORK']:
            NUM_LAYERS=int(config['NETWORK']['num_layers'])
        if 'layer_density' in config['NETWORK']:
            LAYER_DENSITY=int(config['NETWORK']['layer_density'])
        
        if 'num_epochs' in config['TRAINING']:
            NUM_EPOCHS=int(config['TRAINING']['num_epochs'])
        if 'batch_size' in config['TRAINING']:
            BATCH_SIZE=int(config['TRAINING']['batch_size'])
        if 'learning_rate' in config['TRAINING']:
            LEARNING_RATE=float(config['TRAINING']['learning_rate'])
        if 'scale_factor' in config['TRAINING']:
            SCALE_FACTOR=float(config['TRAINING']['scale_factor'])
        if 'patience' in config['TRAINING']:
            PATIENCE=int(config['TRAINING']['patience'])
    
    config['NETWORK']={}
    config['TRAINING']={}
    config['NETWORK']['num_layers']=str(NUM_LAYERS)
    config['NETWORK']['layer_density']=str(LAYER_DENSITY)
    config['TRAINING']['num_epochs']=str(NUM_EPOCHS)
    config['TRAINING']['batch_size']=str(BATCH_SIZE)
    config['TRAINING']['learning_rate']=str(LEARNING_RATE)
    config['TRAINING']['scale_factor']=str(SCALE_FACTOR)
    config['TRAINING']['patience']=str(PATIENCE)
    config.write(open(config_file, "w"))
    
    data=SkinColorDataset('data/skincolor/skin.csv')
    generator=torch.Generator().manual_seed(42)
    train_data, test_data=random_split(data, [0.9, 0.1], generator)
    train_loader=DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader=DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model=SkinColorSegmenterNetwork(num_layers=NUM_LAYERS, layer_density=LAYER_DENSITY)
    print('Network structure:')
    print(model)
    print('Trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(device)

    loss_fn=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=SCALE_FACTOR, patience=PATIENCE)

    start=time.time()
    val_loss=-1
    best_val_loss=1
    best_params=model.state_dict()

    for i in range(NUM_EPOCHS):
        print('Epoch #%d' % (i+1))
        train_loss, train_accuracy=train_loop(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy=test_loop(model, test_loader, loss_fn)
        writer.add_scalars('loss', {'train': train_loss, 'validation': val_loss}, i)
        writer.add_scalars('accuracy', {'train': train_accuracy, 'validation': val_accuracy}, i)
        print('Epoch done. Time since start of training:', time.time()-start)

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            best_params=model.state_dict()
        
        if scheduler:
            scheduler.step(val_loss)
            print('New lr:', scheduler.get_last_lr())

    end=time.time()
    print('Training finished, saving model...')
    torch.save(best_params, Path(path).joinpath('model.pth'))

    model.train()
    model_summary=str(summary(model=model, input_size=np.shape(next(iter(train_loader))[0]), col_width=60, verbose=0)).replace(' ', '&nbsp;').replace('\n', '<br/>')
    writer.add_text('model info',
    f"""data info: RGB pixel values converted to [0, 1] float arrays.<br/>
    train dataset size: {len(train_data)}<br/>
    validation dataset size: {len(test_data)}<br/><br/>
    num_epochs={NUM_EPOCHS}<br/>
    batch_size={BATCH_SIZE}<br/>
    initial_learning_rate={LEARNING_RATE}<br/>
    final_learning_rate={scheduler.get_last_lr()[0] if scheduler else LEARNING_RATE}<br/><br/>
    trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}<br/>
    training time/epoch: {(end-start)/NUM_EPOCHS}<br/>
    Final MSE val. loss: {best_val_loss}<br/><br/>
    Network structure:<br/>
    {model_summary}"""
    )
    writer.add_graph(model, next(iter(test_loader))[0].to(device))
    writer.flush()
    writer.close()
    print('Done!')

def main():
    parser = argparse.ArgumentParser(
        description='Train the network performing the skin color based segmentation',
        epilog="""
Example:
  python scripts/train/skin_color_segmenter/train_model.py --path models/skin_segmentation/model1
        """
    )

    parser.add_argument(
        '--path',
        type=str,
        help='Folder with optional config.txt file and where the model params should be saved'
    )

    args = parser.parse_args()
    if not args.path:
        raise Exception('Path must be given')
    train(args.path)

if __name__ == '__main__':
    main()