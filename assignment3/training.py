"""Loss, training, and evaluation for Assignment 3."""

import math

import torch
from torch.nn import functional as F


def compute_loss(logits, targets):
    """Return the mean cross-entropy loss of the next-token predictions.

    args:
        logits: shape (batch_size, T, vocab_size)
        targets: shape (batch_size, T), torch.long, the next token at every position
    outputs:
        loss: a scalar
    """
    loss = None
    ####################################
    # Your implementation starts here. #
    ####################################
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    ##################################
    # Your implementation ends here. #
    ##################################
    return loss


def evaluate(model, data_loader, device, max_batches=None):
    """Return the mean loss and the perplexity of the model on a dataset.

    args:
        model: the model to evaluate
        data_loader: torch.utils.data.DataLoader that returns batches (x, y)
        device: torch.device, the device to evaluate on
        max_batches: if not None, only evaluate the first max_batches batches
    outputs:
        mean_loss: float, the loss averaged over all evaluated windows
        perplexity: float, exp(mean_loss)
    """
    mean_loss, perplexity = None, None
    ####################################
    # Your implementation starts here. #
    ####################################
    total_loss = 0.0
    num_windows = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = compute_loss(logits, y)

            batch_windows = x.shape[0]
            total_loss += loss.item() * batch_windows
            num_windows += batch_windows

    mean_loss = total_loss / num_windows
    perplexity = math.exp(mean_loss)
    ##################################
    # Your implementation ends here. #
    ##################################
    return mean_loss, perplexity


def train_model(model, train_loader, val_loader, num_epochs, device, eval_every=500, **kwargs):
    """Train the model and record the training and validation losses.

    args:
        model: GPT, the model to train
        train_loader: torch.utils.data.DataLoader for the training set
        val_loader: torch.utils.data.DataLoader for the validation set
        num_epochs: int, the number of epochs to train
        device: torch.device, the device to train on
        eval_every: int, evaluate on the first 50 validation batches every eval_every steps
        **kwargs: include any optional keyword argument that may be helpful, which can include:
            - optimizer_kwargs: arguments for the optimizer
            - scheduler_kwargs: arguments for the learning rate scheduler
    outputs:
        history: a dictionary with
            "train_loss": a list of Python floats, the training loss of every step,
            "val_step": a list of the steps at which the model was evaluated,
            "val_loss": a list of Python floats, the validation loss at these steps
    """
    history = {"train_loss": [], "val_step": [], "val_loss": []}
    ####################################
    # Your implementation starts here. #
    ####################################
    optimizer_kwargs = {"lr": 1e-3, "weight_decay": 0.1}
    optimizer_kwargs.update(kwargs.get("optimizer_kwargs", {}))
    scheduler_kwargs = {"max_lr": 1e-3, "pct_start": 0.05}
    scheduler_kwargs.update(kwargs.get("scheduler_kwargs", {}))

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        total_steps=num_epochs * len(train_loader),
        **scheduler_kwargs,
    )

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = compute_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            history["train_loss"].append(loss.item())
            step += 1

            if step % eval_every == 0:
                val_loss, _ = evaluate(model, val_loader, device, max_batches=50)
                history["val_step"].append(step)
                history["val_loss"].append(val_loss)
                model.train()
    ##################################
    # Your implementation ends here. #
    ##################################
    return history
