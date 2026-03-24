FROM debian:stable

WORKDIR /agent

# Base requirements
RUN apt-get update && apt install -y python3 python-is-python3 python3-pydantic python3-requests

# Additional tools & config
RUN apt install -y procps build-essential git python3-pip file man-db manpages
COPY agent-gitconfig /etc/gitconfig

COPY agent.py agent.toml ./
