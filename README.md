# README

## Files

This repository includes a mock backend server, a mock data source and a modified version of Python script by Arunodhayan Sampath Kumar.

The required files for deployment are under `/deployment`

## Deploy server

in `/deployment/`: 

- move frontend build into `/deployment/static/`
- Modify reverse proxy settings, use correct host names. Currently, it's using Traefik as a reverse proxy, which is not included in this repo. 
- Modify hostname for static hosting server in `/deployment/nginx/config/nginx.conf`
- Modify frontend proxy in `/deployment/main.js` in line 357, pointing it to the static hosting server
- Modify `domina` and `root_url` in `/deployment/grafana/grafana.ini`
- Configure passwords for the MQTT server in `/deployment/config/password.txt` and use them in `/deployment/.env`
- run `docker compose up -d`

## Deplay clients
Mock clients are included in `/deployment/docker-compose.yml`, remove them in production.

in `/py`,
- place the model parameter files in the correct path
- configure username, password, and topic for mqtt client from line 51 to 56. 
- run the Python script