FROM nvidia/cuda:12.6.3-devel-ubuntu22.04
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y python3 python3-pip python3-venv python3-dev python3-wheel g++ git cmake make patch curl nox \
  && curl https://github.com/cli/cli/releases/download/v2.39.1/gh_2.39.1_linux_amd64.deb -L -o /tmp/gh.deb \
  && apt-get install -y /tmp/gh.deb \
  && rm -rf /var/lib/apt/lists/*

COPY cuda-tests-entrypoint /app/entrypoint
ENTRYPOINT ["/app/entrypoint"]
