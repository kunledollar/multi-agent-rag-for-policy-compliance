#!/bin/bash
set -e

cd /home/ubuntu/Sentinel
git pull origin main
docker compose down
docker compose build
docker compose up -d
