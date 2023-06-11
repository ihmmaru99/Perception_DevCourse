from ast import parse
import os, sys
import torch
from torch.utils.data.dataloader import DataLoader
import argparse # cmd에 실행 시 arg parsing

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transforms import *
from model.yolov3 import *
from train.trainer import *



def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH arguments")
    parser.add_argument("--gpus", type = int, help = "List of GPU device id", default = [], nargs='+')
    parser.add_argument("--mode", type = str, help = "mode: train / eval / demo", default = None)
    parser.add_argument("--cfg", type = str, help = "model config path", default = None)
    parser.add_argument("--checkpoint", type = str, help = "model checkpoint path", default = None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    
    # skip invalid data
    if len(batch) == 0:
        return
    
    imgs, targets, anno_path = list(zip(*batch))

    imgs = torch.stack([img for img in imgs])

    for i, boxes in enumerate(targets):
        # insert index of batch
        boxes[:, 0] = i
        print(boxes)
    targets = torch.cat(targets, 0)

    return imgs, targets, anno_path

def train(cfg_param = None, using_gpus = None):
    print("train")
    # data loader
    my_transform = get_transformations(cfg_param=cfg_param, is_train = True)
    train_data = Yolodata(is_train = True,
                        transform = my_transform,
                        cfg_param = cfg_param)
    train_loader = DataLoader(train_data,
                              batch_size = cfg_param['batch'],
                              num_workers = 0,
                              pin_memory = True,
                              drop_last = True, # 6081개의 이미지를 batch_size = 4로 학습시킬 때 1개가 남을 시 다음 epoch에 포함시킬지 여부
                              shuffle = True,
                              collate_fn = collate_fn)   # __getitem__으로부터 받은 image를 batch_size 수만큼 묶어줌

    model = Darknet53(args.cfg, cfg_param, training = True)
    model.train()
    model.initialize_weights()

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print("device : ", device)
    model = model.to(device)
    trainer = Trainer(model = model, train_loader = train_loader, eval_loader = None, hparam = cfg_param, device = device)
    trainer.run()

def eval(cfg_param = None, using_gpus = None):
    print("evaluation")

def demo(cfg_param = None, using_gpus = None):
    print("demo")

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    args = parse_args()

    # cfg parser
    net_data = parse_hyperparm_config(args.cfg)
    print(net_data)

    cfg_param = get_hyperparam(net_data)
    print(cfg_param)

    using_gpus = [int(g) for g in args.gpus]
    # training
    if args.mode == "train":
        train(cfg_param)
    
    # evaluation
    elif args.mode == "eval":
        eval(cfg_param)

    # demo
    elif args.mode == "demo":
        demo(cfg_param)

    else:
        print("Unknown mode")
    
    print("Finish")