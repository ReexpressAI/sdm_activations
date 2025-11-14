# Copyright Reexpress AI, Inc. All rights reserved.

# Note: The core of these classes (DiscVBLLMLP and GenVBLLMLP) is from https://github.com/VectorInstitute/vbll via
# an MIT license Copyright (c) 2024 Vector Institute. Our additions are the helper functions normalize_embeddings() and
# export_properties_to_dict().

import vbll

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import constants

class DiscVBLLMLP(nn.Module):
    def __init__(self, cfg, training_embedding_summary_stats=None):
        super(DiscVBLLMLP, self).__init__()

        self.params = nn.ModuleDict({
            'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
            'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
            'out_layer': vbll.DiscClassification(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES, cfg.REG_WEIGHT, parameterization = cfg.PARAM, return_ood=cfg.RETURN_OOD, prior_scale=cfg.PRIOR_SCALE),
        })
        self.activations = nn.ModuleList([nn.ELU() for i in range(cfg.NUM_LAYERS)])
        self.cfg = cfg
        self.training_embedding_summary_stats = training_embedding_summary_stats

    def normalize_embeddings(self, embeddings):
        # (optional) mean centering of the input embeddings:
        return (embeddings - self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean]) / \
            self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std]

    def forward(self, x, normalize_embeddings=True):
        # global norm
        if normalize_embeddings:
            with torch.no_grad():
                x = \
                    self.normalize_embeddings(x)
        x = x.view(x.shape[0], -1)
        x = self.params['in_layer'](x)

        for layer, ac in zip(self.params['core'], self.activations):
            x = ac(layer(x))

        return self.params['out_layer'](x)

    def export_properties_to_dict(self):
        json_dict = {"IN_FEATURES": self.cfg.IN_FEATURES,
                     "HIDDEN_FEATURES": self.cfg.HIDDEN_FEATURES,
                     "OUT_FEATURES": self.cfg.OUT_FEATURES,
                     "NUM_LAYERS": self.cfg.NUM_LAYERS,
                     "REG_WEIGHT": self.cfg.REG_WEIGHT,
                     "PARAM": self.cfg.PARAM,
                     "RETURN_OOD": self.cfg.RETURN_OOD,
                     "PRIOR_SCALE": self.cfg.PRIOR_SCALE,
                     constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats:
                         self.training_embedding_summary_stats
                     }
        return json_dict


class GenVBLLMLP(nn.Module):
    def __init__(self, cfg, training_embedding_summary_stats=None):
        super(GenVBLLMLP, self).__init__()

        self.params = nn.ModuleDict({
            'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
            'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
            'out_layer': vbll.GenClassification(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES, cfg.REG_WEIGHT, parameterization = cfg.PARAM, return_ood=cfg.RETURN_OOD, prior_scale=cfg.PRIOR_SCALE),
        })
        self.activations = nn.ModuleList([nn.ELU() for i in range(cfg.NUM_LAYERS)])
        self.cfg = cfg
        self.training_embedding_summary_stats = training_embedding_summary_stats

    def normalize_embeddings(self, embeddings):
        # (optional) mean centering of the input embeddings:
        return (embeddings - self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean]) / \
            self.training_embedding_summary_stats[constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std]

    def forward(self, x, normalize_embeddings=True):
        # global norm
        if normalize_embeddings:
            with torch.no_grad():
                x = \
                    self.normalize_embeddings(x)
        x = x.view(x.shape[0], -1)
        x = self.params['in_layer'](x)

        for layer, ac in zip(self.params['core'], self.activations):
            x = ac(layer(x))

        return self.params['out_layer'](x)

    def export_properties_to_dict(self):
        json_dict = {"IN_FEATURES": self.cfg.IN_FEATURES,
                     "HIDDEN_FEATURES": self.cfg.HIDDEN_FEATURES,
                     "OUT_FEATURES": self.cfg.OUT_FEATURES,
                     "NUM_LAYERS": self.cfg.NUM_LAYERS,
                     "REG_WEIGHT": self.cfg.REG_WEIGHT,
                     "PARAM": self.cfg.PARAM,
                     "RETURN_OOD": self.cfg.RETURN_OOD,
                     "PRIOR_SCALE": self.cfg.PRIOR_SCALE,
                     constants.STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats:
                         self.training_embedding_summary_stats
                     }
        return json_dict