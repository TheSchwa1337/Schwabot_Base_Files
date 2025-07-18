#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot GUI Package
====================

GUI components and visualizers for the Schwabot trading system.
"""

from .visualizer_launcher import VisualizerLauncher
from .flask_app import app

__all__ = ['VisualizerLauncher', 'app']
__version__ = '1.0.0' 