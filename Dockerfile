FROM python:latest

# set working directory
WORKDIR /app

# install depenedencies
RUN pip install pipenv
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --deploy

# Copy scripts to folder
COPY . /app