#!/usr/bin/env python3
"""Hal-abuur Red Kinetic engine. Usage: python3 run.py doctor | auto VIDEO SRT | fx | check plan.json | stills plan.json | render plan.json -o out.mp4"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rk.__main__ import main
sys.exit(main())
