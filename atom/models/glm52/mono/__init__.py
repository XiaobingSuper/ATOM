# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-native GLM-5.2 mono contracts and exact-width layer builds.

Production dispatch remains on ``atom.models.glm52_mono`` until an exact width
passes compile-resource, TP4 numerical, and model-level performance gates.
"""
