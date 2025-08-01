'''Train CIFAR10 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import json
from training import train, test, Steps, TrainingState, TrainingResults
from utils import select_model, get_config_value, save_to_csv, get_bs_eps_scheduler, get_lr_scheduler
from utils.data import DynamicBatchSampler
from optim.sgd import SGD

TOTAL_STEPS = None
MAX_BS = 4096


# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with Schedulers')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    epochs = get_config_value(config, "epochs")
    csv_path = config.get("csv_path", "../result/csv/")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Dataset Preparation
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    trainloader_full = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Device Setting
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = get_config_value(config, "model")
    model = select_model(model_name=model_name, num_classes=10).to(device)
    print(f"model: {model_name}")

    criterion = nn.CrossEntropyLoss()

    lr = get_config_value(config, "lr")
    optimizer = SGD(model.parameters(), lr=lr)
    bs_scheduler, bs_step_type = get_bs_eps_scheduler(config, max_bs=MAX_BS)
    lr_scheduler, lr_step_type = get_lr_scheduler(optimizer, config)
    print(optimizer)

    sampler = DynamicBatchSampler(len(trainset), scheduler=bs_scheduler, shuffle=True, drop_last=True)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_sampler=sampler,
        num_workers=16,
        pin_memory=True
    )

    state = TrainingState(
        model=model,
        optimizer=optimizer,
        device=device,
        steps=Steps(),
        lr_scheduler=lr_scheduler,
        lr_step_type=lr_step_type,
        bs_scheduler=bs_scheduler,
        bs_step_type=bs_step_type,
        criterion=criterion,
        epoch=0,
        eps=config.get("eps", 0.1)
    )
    results = TrainingResults()

    for epoch in range(state.epoch, epochs):
        state.epoch = epoch

        if state.bs_step_type == 'periodic' and epoch in [40, 80, 120, 160]:
            state.bs_scheduler.step()
            print(f"bs = {state.bs_scheduler.get_batch_size()}")

        if state.lr_step_type == 'periodic' and epoch in [40, 80, 120, 160]:
            state.lr_scheduler.step()
            lr = state.lr_scheduler.get_last_lr()[0]

        train(state, results, trainloader, trainloader_full, total_steps=TOTAL_STEPS, max_bs=MAX_BS)

        test(state, results, testloader)

        print(f'Epoch: {epoch + 1}, Steps: {state.steps.total}, Train Loss: {results.train[epoch][2]:.4f}, Test Acc: {results.test[epoch][3]:.2f}%')

    # Save to CSV file
    save_to_csv(csv_path, results)
