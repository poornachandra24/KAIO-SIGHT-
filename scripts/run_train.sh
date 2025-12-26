#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/training/trainer.py
