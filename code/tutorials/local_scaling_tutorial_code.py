# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import numpy as np

# Tip: Drop this into a console and try some values to gain intuition on the re-scaling (as, for example,
# compared to softmax):

def local_soft_sdm_max(batch_input, q, distance_quantile_per_class=None, log=False, change_of_base=True):
    assert len(batch_input.shape) == 2
    assert batch_input.shape[0] == q.shape[0]
    assert q.shape[1] == 1
    if distance_quantile_per_class is not None:
        assert batch_input.shape == distance_quantile_per_class.shape
    q_rescale_offset = 2
    q_factor = q_rescale_offset + q
    # The offset here assumes every column of d for a given batch (i.e., row) is the same value, as is standard with
    # SDM activations:
    batch_input = batch_input - torch.amax(batch_input, dim=1, keepdim=True)  # for numerical stability
    if distance_quantile_per_class is not None:
        rescaled_distribution = q_factor ** (batch_input * distance_quantile_per_class)
    else:
        rescaled_distribution = q_factor ** batch_input
    if log:  # log_base{q}
        kEPS = torch.finfo(torch.float32).eps  # adjust as applicable for platform
        rescaled_distribution = torch.log(rescaled_distribution + kEPS) - torch.log(
            torch.sum(rescaled_distribution, dim=1) + kEPS).unsqueeze(1)
        if change_of_base:
            return rescaled_distribution / torch.log(q_factor)
        else:
            return rescaled_distribution
    else:
        return rescaled_distribution / torch.sum(rescaled_distribution, dim=1).unsqueeze(1)


# Tip: As a reminder, when converting from log space with q as the base, we need to take the exponent
# with q as the base to recover the probability space. In our case, we also need to take into account the
# rescale offset, for which we provide a convenience function in sdm_model. Here's a local version:

def local_soft_sdm_max_log_to_probability(batch_input, q):
    assert len(batch_input.shape) == 2
    assert batch_input.shape[0] == q.shape[0]
    assert q.shape[1] == 1
    q_rescale_offset = 2
    q_factor = q_rescale_offset + q
    return q_factor ** batch_input


def local_soft_sdm_max_torch_equivalent(batch_input, q, distance_quantile_per_class=None, log=False,
                                        change_of_base=True):
    """
        Alternatively, we can avoid reimplementing numerical stability code and instead use standard softmax
        in existing frameworks by using the relation (2+q)^(z'*d) == e^(z'*d*log_e(2+q)), given that 2+q > 0.
        In this case, it is still necessary to rescale the log in the
        loss (or as for example, (cross-entropy loss per instance) / log_e(2+q)) to multiply the loss by
        1/log_e(2+q) relative to the rescaled CE loss. See also local_soft_sdm_max_torch_equivalent_streamlined().
    """
    assert len(batch_input.shape) == 2
    assert batch_input.shape[0] == q.shape[0]
    assert q.shape[1] == 1
    if distance_quantile_per_class is not None:
        assert batch_input.shape == distance_quantile_per_class.shape
    q_rescale_offset = 2
    q_factor = q_rescale_offset + q
    log_q_factor = torch.log(q_factor)
    if distance_quantile_per_class is not None:
        scaled_logits = batch_input * distance_quantile_per_class * log_q_factor
    else:
        scaled_logits = batch_input * log_q_factor
    if log:
        log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)  # log_softmax for numerical stability
        if change_of_base:
            return log_probs / log_q_factor
        else:
            return log_probs
    else:
        return torch.softmax(scaled_logits, dim=-1)


# For reference:
def local_soft_sdm_max_torch_equivalent_streamlined(batch_input, q, distance_quantile_per_class=None,
                                                    return_scaled_logits_and_log_q_factor=False):
    """
        This uses the relation (2+q)^(z'*d) == e^(z'*d*log_e(2+q)), given that 2+q > 0, in order to use existing
        optimized softmax and CE loss implementations.

        If return_scaled_logits=True, this can be used with the standard CE loss after scaling by log_q_factor:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            scaled_logits, log_q_factor = local_soft_sdm_max_torch_equivalent_streamlined(
                batch_z_prime, batch_q, distance_quantile_per_class=batch_d,
                return_scaled_logits=True
            )
            # Note the .squeeze() is necessary for the correct broadcast:
            loss_for_batch = loss_fn(scaled_logits, targets) / log_q_factor.squeeze()
            loss_for_batch.mean().backward()  # Note the .mean()
            # Because an absence of .squeeze() could be a source of subtle bugs, we return log_q_factor.squeeze() here.
    """
    assert len(batch_input.shape) == 2
    assert batch_input.shape[0] == q.shape[0]
    assert q.shape[1] == 1
    if distance_quantile_per_class is not None:
        assert batch_input.shape == distance_quantile_per_class.shape
    q_rescale_offset = 2
    q_factor = q_rescale_offset + q
    log_q_factor = torch.log(q_factor)
    if distance_quantile_per_class is not None:
        scaled_logits = batch_input * distance_quantile_per_class * log_q_factor
    else:
        scaled_logits = batch_input * log_q_factor
    if return_scaled_logits_and_log_q_factor:
        return scaled_logits, log_q_factor.squeeze()
    else:
        return torch.softmax(scaled_logits, dim=-1)
