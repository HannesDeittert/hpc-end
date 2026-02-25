# Docker (full stEVE stack)

This image is the **recommended** way to run long training jobs:
- prebuilt SOFA binaries (fast + stable)
- pinned stEVE repos (reproducible)
- GPU-ready PyTorch
- includes `stEVE_training` and this repo

If you need a custom SOFA build, see `third_party/stEVE_training/dockerfile` (builds SOFA from source, slower and more brittle).

## Build

```bash
docker build -f docker/Dockerfile -t steve-training .
```

## Run (GPU training)

```bash
docker run --gpus all --ipc=host --shm-size 15G -it \
  -v /path/to/results:/workspace/results \
  -v /path/to/data:/workspace/steve_recommender/data \
  steve-training
```

## GPU prerequisites (host)

If you see `could not select device driver "" with capabilities: [[gpu]]`, your host
is missing the NVIDIA Container Toolkit (or drivers).

Quick check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

Install the toolkit (Ubuntu / Debian):

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Inside the container

Run this repo's trainer:

```bash
python3 -m steve_recommender.rl.train_paper_arch --tool <model/wire> --device cuda --workers 8
```

Or the upstream scripts:

```bash
cd /opt/eve_training
python3 ./training_scripts/BasicWireNav_train.py -d cuda -nw 8 -lr 0.0002 --hidden 900 900 900 900 -en 500 -el 1 -n BasicWireNav
```

## Smoke test (CPU or GPU)

CPU (works without NVIDIA drivers):

```bash
python3 -m steve_recommender.rl.smoke_train \
  --tool TestModel_StandardJ035/StandardJ035_PTFE \
  --device cpu
```

GPU:

```bash
python3 -m steve_recommender.rl.smoke_train \
  --tool TestModel_StandardJ035/StandardJ035_PTFE \
  --device cuda
```

## Notes

- Base: Ubuntu 20.04 + CUDA 11.8 runtime + SOFA v23.06.00 binaries.
- Results should be mounted to `/workspace/results` to persist outside the container.
- If you want to reuse your local tools, mount `data/` into `/workspace/steve_recommender/data`.
- UI usage inside Docker requires X11/Wayland forwarding; for training, a headless container is recommended.

## Process management (start/stop/restart/terminate)

Run a long training job in the background with a stable container name:

```bash
docker run --name steve-train --gpus all --ipc=host --shm-size 15G -d \
  -v /path/to/results:/workspace/results \
  -v /path/to/data:/workspace/steve_recommender/data \
  steve-training \
  python3 -m steve_recommender.rl.train_paper_arch --tool <model/wire> --device cuda --workers 8
```

Check status + logs:

```bash
docker ps -a | rg steve-train
docker logs -f steve-train
```

Stop / restart / terminate:

```bash
docker stop steve-train      # graceful stop (SIGTERM)
docker start steve-train     # restart the same container
docker restart steve-train   # stop + start
docker kill steve-train      # force stop (SIGKILL)
```

Enter the container shell:

```bash
docker exec -it steve-train bash
```

Remove the container after stopping:

```bash
docker rm steve-train
```

Tip: If you use `--rm`, the container is deleted on exit, so it cannot be restarted.
