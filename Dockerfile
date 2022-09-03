FROM python:latest

# set working directory
WORKDIR /app

# install depenedencies
RUN pip install pipenv
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --system --deploy --ignore-pipfile

# Copy scripts to folder
COPY . /app

# start the program
# CMD ["python", "app.py"]