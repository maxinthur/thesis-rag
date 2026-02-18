#!/bin/bash
pip install -r /home/site/wwwroot/requirements.txt
python -m streamlit run /home/site/wwwroot/app.py --server.port 8000 --server.address 0.0.0.0
