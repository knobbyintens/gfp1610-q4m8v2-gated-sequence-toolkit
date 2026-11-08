"""Sparsity regularisers used across the toolkit."""

import torch


def l0_approx(mask, eps=1e-8):
    return (mask / (mask + eps)).sum()
