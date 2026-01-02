#!/bin/bash
pkg update -y
pkg upgrade -y
pkg install python git -y
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt