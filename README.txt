INFOMDDS - Dashboard

This folder holds the dashboard we have create for CrimeStat.

The folder contains several files and subfolders:

- /notebook > Holds the development code used, not used for the actual process.
- /dashboard > This is a Flask Dash implementatation of the project.
    - /pages: Contains the individual pages of the app, that show and handle the data. Every page ahs its own file.
    - app.py > This is the main Flask app that hosts the dashboard.
    - Dockerfile_dashboard > Dockerfile for the flask-app.
    - requirements.txt > Holds all python libraries that are used.
- docker-compose.yml > Is used to set up the environments and docker containers.
- import_data.py > contains the actual import of the files to the database

To run:
- Open a terminal or command prompt
- CD to this directory called 'INFOMDSS_Dashboard'
- run 'docker compose up'. In some versions, you might need docker-compose up
- You can also use docker-compose up -d for running in daemon mode (hidden)
- Running this command for the first time might take a few minutes, as all the data is imported.
- Jupyter notebook is available at localhost:8888 and dashboard at localhost:8080