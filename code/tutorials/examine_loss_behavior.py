# Copyright Reexpress AI, Inc. All rights reserved.

"""
Example code for the tutorial examining the loss and the gradient.
"""
import argparse
import torch.nn as nn
import torch

from local_scaling_tutorial_code import local_soft_sdm_max_torch_equivalent as soft_sdm_max


def compare_loss_and_gradient_behavior(logits, target_class, distance_quantiles, q=None):
    """
    logits: raw scores before final sdm/softmax, shape [1, 2]
    target_class: integer label (0 or 1)
    distance_quantiles: list of floats in [0,1]
    q: list of integers|floats in [0, inf)

    """
    if q is None:
        ce_loss = nn.CrossEntropyLoss()
    else:
        nll_loss = nn.NLLLoss(reduction='mean')
    q_vals = [1] if q is None else q
    for q_val in q_vals:
        for d in distance_quantiles:
            target = torch.tensor([target_class])
            # Scale logits by d (or equivalently, instance-wise inverse temperature).
            # Here we keep it simple for illustrative purposes. Alternatively, add a linear layer or other simple
            # structure. Note that q and d (as with the inverse temperature) are not learned parameters updated
            # directly by SGD: q is a depth-wise match into the support set and d is a quantile from the class-wise
            # empirical CDF. (As such, by design, this is a different mechanism than back-propagating through
            # bi-encoder/cross-encoder search, which has a different use case. See my work from 2020. In such
            # architectures, the SDM activation can be used as the final layer.)
            scaled_logits = (d * logits).clone().requires_grad_(True)
            if q is None:
                # Compute loss
                offset = "\t"
                loss = ce_loss(scaled_logits, target)
            else:
                offset = ""
                loss = nll_loss(soft_sdm_max(scaled_logits, torch.tensor([q_val]).unsqueeze(1), log=True,
                                             change_of_base=True),
                                target)
            loss.backward()
            # Gradient wrt logits
            grad = scaled_logits.grad.detach().squeeze()
            grad_norm = torch.linalg.vector_norm(grad, ord=2).item()

            if q is None:
                print(f"1/T = {d:.2f}")
                print(f"  softmax = {torch.softmax(d * logits, dim=-1)}")
            else:
                print(f"q = {q_val}")
                print(f"{offset}  d = {d:.2f}")
                print(f"{offset}  sdm = "
                      f"{soft_sdm_max(scaled_logits, torch.tensor([q_val]).unsqueeze(1), log=False)}")
            # For the SDM loss, the loss is rescaled by a factor of 1/log_e(2+q) relative to standard cross-entropy.
            print(f"{offset}  Loss = {loss.item():.4f}")
            print(f"{offset}  Gradient = {grad.numpy()}")
            print(f"{offset}  Gradient L2 norm = {grad_norm:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="-----[Scaling Tutorial]-----")
    options = parser.parse_args()

    # # Example: logits very strongly favor class 1, but true class is label 0:
    # #  Ref: torch.softmax(torch.tensor([[1.0, 30.0]]), dim=-1): tensor([[2.5437e-13, 1.0000e+00]])
    # logits = torch.tensor([[1.0, 30.0]])  # 'logits' are the raw scores before softmax
    # target_class = 0

    # Example: logits favor class 1, but true class is label 0:
    #  Ref: torch.softmax(torch.tensor([[1.0, 3.0]]), dim=-1): tensor([[0.1192, 0.8808]])
    logits = torch.tensor([[1.0, 3.0]])  # 'logits' are the raw scores before softmax
    target_class = 0

    distance_quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print(f"Cross-entropy with inverse temperature")
    compare_loss_and_gradient_behavior(logits, target_class, distance_quantiles, q=None)

    q = [0, torch.e-2, 1, 100]
    print(f"SDM loss")
    compare_loss_and_gradient_behavior(logits, target_class, distance_quantiles, q=q)


if __name__ == "__main__":
    main()
