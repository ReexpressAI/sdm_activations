# Copyright Reexpress AI, Inc. All rights reserved.

from vbll_model import DiscVBLLMLP
from vbll_model import GenVBLLMLP

import baseline_utils_vbll_training

import logging
import sys
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# train_cfg and cfg are from the VBLL classification tutorial
class train_cfg:
    NUM_EPOCHS = None
    BATCH_SIZE = 50
    LR = 3e-3  # This gets set after class initialization with options.learning_rate
    # In our initial experiments, the approach was not particularly sensitive to the weight-decay parameter,
    # so we keep this the same as in the VBLL classification tutorial. There is no weight decay on the final layer.
    WD = 1e-4
    OPT = torch.optim.AdamW
    CLIP_VAL = 1
    VAL_FREQ = 1
    VBLL = True

class cfg:
    IN_FEATURES = 6144  # This gets set after class initialization.
    HIDDEN_FEATURES = 795  # This gets set after class initialization. approx. 6144 * d + 2*(d*d) + d*2
    OUT_FEATURES = 2  # This gets set after class initialization.
    NUM_LAYERS = 2  # We keep this constant as in the paper.
    REG_WEIGHT = None  # This gets set after class initialization. Default is 1.0/|D_tr|
    PARAM = 'diagonal'
    RETURN_OOD = True
    PRIOR_SCALE = 1.


def train(options, train_embeddings=None, calibration_embeddings=None,
          train_labels=None, calibration_labels=None,
          model_params=None,
          main_device=None, model_dir=None, model=None, shuffle_index=0):
    import baseline_utils_model_vbll

    if model is None:

        model_config = cfg()
        model_config.REG_WEIGHT = 1./train_embeddings.shape[0]
        print(f"VBLL regularization multiplicative factor: {options.vbll_regularization_multiplicative_factor}")
        assert options.vbll_regularization_multiplicative_factor >= 1.0
        model_config.REG_WEIGHT *= options.vbll_regularization_multiplicative_factor
        print(f"model_config.REG_WEIGHT: {model_config.REG_WEIGHT}, which is (1.0/|D_tr|) * regularization factor: "
              f"(1.0/{train_embeddings.shape[0]}) * {options.vbll_regularization_multiplicative_factor}")
        model_config.IN_FEATURES = train_embeddings.shape[1]
        model_config.OUT_FEATURES = options.class_size
        model_config.HIDDEN_FEATURES = options.vbll_hidden_dimension
        print(f"The VBLL model has an input linear layer, 2 core linear layers, and an output linear layer. The "
              f"hidden dimension is {options.vbll_hidden_dimension}.")

        train_config = train_cfg
        train_config.NUM_EPOCHS = options.epoch
        train_config.BATCH_SIZE = options.batch_size
        train_config.LR = options.learning_rate

        if options.is_discriminative_vbll_model:
            print("Initializing DiscVBLLMLP() model")
            model = DiscVBLLMLP(model_config,
                                training_embedding_summary_stats=model_params["training_embedding_summary_stats"]).to(
                main_device)
        elif options.is_generative_vbll_model:
            print("Initializing GenVBLLMLP() model")
            model = GenVBLLMLP(model_config,
                               training_embedding_summary_stats=model_params["training_embedding_summary_stats"]).to(
                main_device)

    # For the baseline adaptor, we use the 'pretrain' routine
    min_held_out_average_loss, min_held_out_average_loss_epoch = \
        baseline_utils_vbll_training.train(options, model=model, model_dir=model_dir,
                                           held_out_embeddings=calibration_embeddings,
                                           held_out_labels=calibration_labels,
                                           train_embeddings=train_embeddings,
                                           train_labels=train_labels,
                                           learning_rate=options.learning_rate,
                                           return_min_held_out_average_loss=True,
                                           main_device=main_device, train_cfg=train_cfg,
                                           shuffle_index=shuffle_index
                                           )
    return min_held_out_average_loss, min_held_out_average_loss_epoch