# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from . import autosummary
from . import network
from . import optimizer
from . import tfutil
from . import custom_ops

from .tfutil import *
from .network import Network

from .optimizer import Optimizer

from .custom_ops import get_plugin

# manage implement.
default_impl = 'cuda'

def set_impl(impl):
    global default_impl
    default_impl = impl

def which_impl(impl):
    global default_impl
    return impl if impl != None else default_impl
