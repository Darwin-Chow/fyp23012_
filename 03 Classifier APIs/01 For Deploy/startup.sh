#!/bin/sh
apt update
apt-get install -y libgomp1
gunicorn app:app