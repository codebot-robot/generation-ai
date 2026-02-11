# gRPC Fine-tuning Experiment

This experiment implements a gRPC-based fine-tuning service using the `trl` library.
It allows offloading heavy fine-tuning dependencies (PyTorch, Transformers, etc.) to a server pod while keeping the client lightweight.

## Structure

- `proto/`: gRPC service definition.
- `src/`: Server and client implementation.
- `images/`: Dockerfiles for server and client.
- `k8s/`: Kubernetes manifests.
- `dev/tasks/`: Helper scripts for running in Kubernetes.

## Running in Kubernetes

You can use the provided script to build images and deploy to GKE:

```bash
./dev/tasks/run-in-kube
```

The script will:
1. Build the server and client images.
2. Push them to your GCR.
3. Deploy the server as a Deployment and Service.
4. Run the client as a Job.
5. Stream the logs from the client.

## Local Testing

To test locally, it is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install trl transformers datasets grpcio grpcio-tools
./generate_proto.sh
python3 src/server.py &
python3 src/client.py
```
