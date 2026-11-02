"""Optional real CIFAR-10 experiment. Requires explicit dataset download.

No pretraining; deliberately small models. Do not compare short-run accuracy
with original ViT checkpoints pretrained on millions of images.
"""
import argparse
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .model import TinyCNN, TinyViT
from .train import set_seed, evaluate


def run(args):
    set_seed(args.seed)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616)),
    ])
    train_all=datasets.CIFAR10(args.data_dir,train=True,download=args.download,transform=train_tf)
    val_all=datasets.CIFAR10(args.data_dir,train=True,download=False,transform=eval_tf)
    test=datasets.CIFAR10(args.data_dir,train=False,download=args.download,transform=eval_tf)
    indices=torch.randperm(len(train_all),generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_idx=indices[:45000][:args.max_train]
    val_idx=indices[45000:]
    assert not set(train_idx).intersection(val_idx)
    train_loader=DataLoader(Subset(train_all,train_idx),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(Subset(val_all,val_idx),batch_size=args.batch_size,num_workers=0)
    test_loader=DataLoader(test,batch_size=args.batch_size,num_workers=0)
    if args.model=='cnn':
        model=TinyCNN(num_classes=10)
    else:
        model=TinyViT(patch_size=4 if args.model=='vit4' else 8,num_classes=10)
    model=model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    best_acc=-1
    best_state=None
    for ep in range(args.epochs):
        model.train()
        ts=time.perf_counter()
        for x,y in train_loader:
            x,y=x.to(args.device),y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x),y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        val=evaluate(model,val_loader,args.device)
        if val['accuracy']>best_acc:
            best_acc=val['accuracy']
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'epoch={ep+1} val_acc={val["accuracy"]:.4f} seconds={time.perf_counter()-ts:.1f}',flush=True)
    model.load_state_dict(best_state)
    result=evaluate(model,test_loader,args.device)
    print(f'FINAL HELD-OUT TEST ACCURACY {result["accuracy"]:.4f}')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['cnn','vit8','vit4'],default='vit8')
    p.add_argument('--data-dir',default='./data')
    p.add_argument('--download',action='store_true',help='Explicitly download CIFAR-10')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--max-train',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    run(args)

if __name__=='__main__': main()
