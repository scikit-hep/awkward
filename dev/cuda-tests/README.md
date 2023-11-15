# CUDA Tests Container

This directory containers resources for running the Awkward CUDA tests inside a Docker container. It is possible to use other container runtimes e.g. podman.

## Build Container

1. Build Container

    Only required if not using the pre-built container image
    ```bash
    docker build -f cuda-tests.Dockerfile -t awkward/cuda-tests:latest .
    ```
2. Install systemd units (optional)
    ```bash
    sudo cp cuda-tests.service cuda-tests.timer /etc/systemd/system/
    ```
3. Activate systemd units (optional)
    ```bash
    sudo systemctl enable cuda-tests.service cuda-tests.timer
    ```
4. Store GitHub API token with `repo` credentials in `/etc/cuda-gh-token`
    ```bash
    sudo echo "ghp_..." > /etc/cuda-gh-token
    ```
5. Run container (if not using systemd)
    ```bash
    docker run --rm \
    --runtime=nvidia \
    --gpus all \
    -v "/etc:/creds" \
    -e GH_TOKEN_PATH=/creds/cuda-gh-token \
    agoose77/cuda-tests:latest
    ```
