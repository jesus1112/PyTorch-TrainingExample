"""
Training of Neural Networks Using PyTorch
In this scrip, a basic neural network is trained from scratch to classify handwritten digits (MNIST dataset)

"""
import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BasicArbitraryCNNmodel

# General parameters
DEVICE = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')  # Detects if there is a cuda capable GPU available
IMG_SIZE = (64, 64)
DATASET_ROOT = './MNIST'
CKPT_PATH = './checkpoint'
LEARNING_RATE = 0.0001
ADAM_BETAS = (0.9, 0.999)
BATCH_SIZE = 256
NUM_EPOCHS = 10
TRAIN_WORKERS = 0
TEST_WORKERS = 0

if __name__ == '__main__':
    """Process:
    1) Create an instance of a Neural Network
    2) Create an instance of a MNIST dataset (train and val) and corresponding dataloaders
    3) Create an instance of an Adam Optimizer and loss criterion
    4) Loop over each epoch
        4.1) Loop over each minibatch
            4.1.1) Move data to DEVICE
            4.1.2) Clear model gradients
            4.1.3) Forward pass
            4.1.4) Calculate loss and save value
            4.1.5) Backward pass (backpropagation of gradients)
            4.1.6) Update model parameters (weights)
        4.2) Loop to evaluate model
            4.2.1) Forward pass the validation data
            4.2.2) Calculate validation error and register it
            4.2.3) Save model weights if the accuracy is higher than before
    5) Save loss values and error values
    """

    # 1) Instance of Neural Network
    net = BasicArbitraryCNNmodel(in_channels=1).to(DEVICE)

    # 2) Instance of MNIST Dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Train data
    train_data = torchvision.datasets.MNIST(root=DATASET_ROOT, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_WORKERS,
    )

    # Test data
    test_data = torchvision.datasets.MNIST(root=DATASET_ROOT, transform=transform, download=True, train=False)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=TRAIN_WORKERS,
    )

    # 3) Instance of Adam Optimizer and loss criterion
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    loss = torch.nn.CrossEntropyLoss()

    # 4) Loop over each epoch
    testErr_list = []
    loss_list = []
    best_accuracy = 0
    best_epoch = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        # 4.1) Loop over each minibatch
        i = 0
        net.train(True)
        iter_loop = tqdm(train_loader, leave=True)
        for inputs, targets in iter_loop:
            # 4.1.1) Move data to DEVICE
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            # 4.1.2) Clear gradients
            opt.zero_grad()
            # 4.1.3) Forward pass
            outputs = net(inputs)
            # 4.1.4) Calculate loss and save value
            loss_value = loss(outputs, targets)
            loss_list.append(loss_value.item())
            # 4.1.5) Backward pass
            loss_value.backward()
            # 4.1.6) Update model parameters
            opt.step()
            i += 1
            if i % (len(train_loader) // 10) == 0:
                iter_loop.set_postfix(loss=loss_value.item())

        # 4.2) Loop to evaluate model
        print('Testing model...')
        net.train(False)
        correct = 0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = net(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct += torch.sum(pred == targets).item()
            acc = correct / len(test_data)
            print(f'Current accuracy: {acc * 100}%')
            testErr_list.append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                if not os.path.exists(CKPT_PATH):
                    os.makedirs(CKPT_PATH)
                torch.save(net.state_dict(), f'{CKPT_PATH}/best_checkpoint.pth')
                print(f'New best accuracy achieved! Checkpoint saved after {epoch + 1} epochs')
    # 5) Saves loss values and error values
    results_dict = {
        'loss_values': loss_list,
        'test_error_values': testErr_list,
    }
    json.dump(results_dict, open(f'./results.json', 'w'))
    print('Training finished')
