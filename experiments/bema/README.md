# Bema - Session-based Chat Server

Bema is a platform for session-based chat, designed to support agentic lifecycles and multi-turn scenarios.

## Key Features

- **Session-based API**: Manage chat sessions as persistent resources.
- **Watch Support**: Get notified of all changes to a session in real-time.
- **Persistence**: Sessions are stored persistently, allowing clients to disconnect and reconnect without losing state.

## Directory Structure

- `proto/`: gRPC service definition.
- `pkg/api/v1alpha1/`: Generated Go code from proto.
- `pkg/server/`: Core server implementation.
- `cmd/bema/`: Server entry point.
- `k8s/`: Kubernetes manifests for deployment.

## Development

### Generate Proto

To regenerate the gRPC code, run:

```bash
./generate_proto.sh
```

### Run Tests

```bash
go test ./...
```

### Local Run

```bash
# Start the server
go run cmd/bema/main.go --storage-dir=/tmp/bema

# In another terminal, run the CLI client
go run cmd/bema-cli/main.go --server=http://localhost:50051
```

## CLI Client

The `bema-cli` supports the following options:

- `--server`: Server address (e.g., `http://localhost:50051`, `https://bema.example.com`).
- `--list`: List all available sessions.
- `--session <id>`: Resume a specific session.

Special support for Kubernetes: if you specify a server URL like `http://bema.default.cluster.svc.local`, the CLI will automatically set up a `kubectl port-forward` to the service.

Inside the CLI, you can use:
- `/session <id>`: Switch to a different session.
- `/quit` or `/exit`: Exit the CLI.
