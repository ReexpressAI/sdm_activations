# Copyright Reexpress AI, Inc. All rights reserved.

import constants
import utils_model

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

import baseline_utils_model_vbll

def train(options, model=None, model_dir=None,
          held_out_embeddings=None,
          held_out_labels=None,
          train_embeddings=None,
          train_labels=None,
          learning_rate=None,
          return_min_held_out_average_loss=False, main_device=None, train_cfg=None, shuffle_index=0):
    device_label = "main_device"
    assert main_device is not None
    current_device = main_device
    total_epochs = options.epoch
    print(f"Training VBLL model for {total_epochs} epochs on {current_device} ({device_label})")
    assert model is not None
    model = model.to(current_device)

    assert train_embeddings is not None
    train_size = train_embeddings.shape[0]

    # for VBLL models, set weight decay to zero on last layer
    param_list = [
        {'params': model.params.in_layer.parameters(), 'weight_decay': train_cfg.WD},
        {'params': model.params.core.parameters(), 'weight_decay': train_cfg.WD},
        {'params': model.params.out_layer.parameters(), 'weight_decay': 0.}
    ]

    # total = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total:,}")
    # exit()
    # Total parameters: 6,154,099 for VBLL vs 6,147,002 for SDM activation with Phi-3.5
    # Total parameters: 7,783,845 for VBLL vs 8,195,002 for SDM activation with Mixtral-8x7b
    print(f"Starting pretraining over {train_size} instances with LR={learning_rate}")
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    optimizer = train_cfg.OPT(param_list,
                              lr=train_cfg.LR,
                              weight_decay=train_cfg.WD)

    all_epoch_cumulative_losses = []

    min_held_out_average_loss = np.inf
    min_held_out_average_loss_epoch = -1

    batch_size = options.batch_size
    for e in range(total_epochs):
        # shuffle data
        shuffled_train_indexes = torch.randperm(train_embeddings.shape[0])
        shuffled_train_embeddings = train_embeddings[shuffled_train_indexes]
        shuffled_train_labels = train_labels[shuffled_train_indexes]
        batch_num = 0
        cumulative_losses = []

        for i in range(0, train_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, train_size - i)

            batch_x = shuffled_train_embeddings[i:i + batch_range].to(current_device)
            batch_y = shuffled_train_labels[i:i + batch_range].to(current_device)
            optimizer.zero_grad()
            model.train()
            out = model(batch_x)
            loss = out.train_loss_fn(batch_y)

            cumulative_losses.append(loss.item())
            loss.backward()
            # "It is strongly recommended that you use gradient clipping with VBLL models." is what is written in the
            # VBLL Regression tutorial, but then the classification tutorial does not actually use it. We include
            # it here since the example training configuration does include CLIP_VAL as a property.
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.CLIP_VAL)
            optimizer.step()

        print(f"---------------Shuffle index: {shuffle_index}. Epoch: {e + 1}---------------")
        print(f"Training Epoch average loss (over-pretraining set): {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Training Average loss across all mini-batches (all epochs, over-pretraining set): "
              f"{np.mean(all_epoch_cumulative_losses)}")

        held_out_marginal_acc, held_out_average_loss = get_loss_over_heldout_data(options, model,
                                                                                  held_out_embeddings,
                                                                                  held_out_labels,
                                                                                  current_device)
        print(f"Training Epoch: {e + 1} / Held-out set marginal loss: {held_out_average_loss}")
        print(f"Training Epoch: {e + 1} / Held-out set marginal accuracy: {held_out_marginal_acc}")

        is_best_running_epoch = held_out_average_loss <= min_held_out_average_loss
        if held_out_average_loss <= min_held_out_average_loss:
            min_held_out_average_loss = held_out_average_loss
            min_held_out_average_loss_epoch = e + 1
        if is_best_running_epoch:
            baseline_utils_model_vbll.save_baseline_model(model, model_dir)
            print(f">Cached best epoch parameters<")
        print(f"\tTraining: Current min held-out set Balanced loss: "
              f"{min_held_out_average_loss} at epoch {min_held_out_average_loss_epoch}")
    print(
        f"\tTraining: Min held-out set Balanced loss: "
        f"{min_held_out_average_loss} at epoch {min_held_out_average_loss_epoch}")

    if return_min_held_out_average_loss:
        return min_held_out_average_loss, min_held_out_average_loss_epoch


def eval_acc(preds, y):
  map_preds = torch.argmax(preds, dim=1)
  return (map_preds == y).float().mean()


def get_loss_over_heldout_data(options, model,
                               held_out_embeddings,
                               held_out_labels,
                               current_device):

    batch_size = options.batch_size
    held_out_size = held_out_embeddings.shape[0]
    batch_num = 0
    model.eval()

    # Here we keep it simple, since VBLL does not use balanced loss
    running_val_loss = []
    running_val_acc = []

    with torch.no_grad():
        for i in range(0, held_out_size, batch_size):
            batch_num += 1
            batch_range = min(batch_size, held_out_size - i)

            batch_x = held_out_embeddings[i:i + batch_range].to(current_device)
            batch_y = held_out_labels[i:i + batch_range].to(current_device)
            out = model(batch_x)
            loss = out.val_loss_fn(batch_y)
            probs = out.predictive.probs
            acc = eval_acc(probs, batch_y).item()

            running_val_loss.append(loss.item())
            running_val_acc.append(acc)

    marginal_acc = np.mean(running_val_acc)
    average_loss = np.mean(running_val_loss).item()

    return marginal_acc, average_loss
