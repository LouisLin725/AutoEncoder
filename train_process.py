import numpy as np
import torch
from tqdm import tqdm


def train(model, n_epochs, train_loader, optimizer, criterion, batch_size, scheduler, ckpt_name):
    """
    Args:
        model (input): loading the model
        n_epochs (int): number of epoch
        train_loader: training dataset
        optimizer (torch.optm): predefined optimizer
        criterion (loss.fnc): predefined loss function
        batch_size (int): size of batch
        scheduler (lr.schedule): learning schedule
        ckpt_name: checkpoint name

    Returns:
        train_losses_his: record the training losses
        model: trained model
    """
    # Storage variable declaration
    train_losses_his = []

    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_losses = []
        print('running epoch: {}'.format(epoch))

        # train the model #
        model.train()

        for data, target in tqdm(train_loader):
            # flatten
            inputs = data.view(-1, 784)

            # initialization
            model.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            codes, decodes = model(inputs)

            # calculate the batch loss, criterion
            loss = criterion(decodes, inputs)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_losses.append(loss.item() * data.size(0))

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

        # calculate average losses
        train_loss = np.average(train_losses)
        train_losses_his.append(train_loss)

        scheduler.step()
        # print training/validation statistics
        print('\tTraining Loss: {:.6f}'.format(train_loss))

        # Setting Check point
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    'train_loss': train_loss,
                    }, ckpt_name)

    return train_losses_his, model
