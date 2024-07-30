#!/bin/bash

# --------- Setup ---------
# Clone the repository for H2O LLM evaluation
# git clone https://github.com/h2oai/h2o-LLM-eval.git
# Navigate to the repository directory
# cd h2o-LLM-eval

# --------- Set Up Database ---------
# Create a Docker volume for the database
docker volume create llm-eval-db-data

# Run a PostgreSQL container with the created volume and a password
docker run -d --name=llm-eval-db -p 5432:5432 \
    -v llm-eval-db-data:/var/lib/postgresql/data \
    -e POSTGRES_PASSWORD=pgpassword postgres:14

# --------- Run Docker Compose ---------
# Start all services defined in the Docker Compose file
docker compose up -d

# --------- Initialize Database ---------
# Install/update pip and the required Python packages
# pip install --upgrade pip
# pip install -r requirements.txt

# Initialize the database schema using an SQL file
PGPASSWORD=pgpassword psql --host=localhost --port=5432 --username=postgres < data/10_init.sql

# --------- Run the Application ---------
# Set environment variables and run the H2O Wave server for the LLM evaluation application
POSTGRES_HOST=localhost POSTGRES_USER=maker POSTGRES_PASSWORD=makerpassword POSTGRES_DB=llm_eval_db H2O_WAVE_NO_LOG=true wave run llm_eval/app.py

# Open your browser and navigate to http://localhost:10101/ to access the application
