# AgentFS - Ephemeral CSI Driver

AgentFS is an experimental CSI driver that provides a simple ephemeral filesystem for agents.

## Vision

The goal of AgentFS is to provide a filesystem that is ephemeral by default but can be periodically snapshotted. This allows agents to have a persistent state at specific points in time without the overhead of a fully distributed POSIX filesystem.

## Status

This is currently in the early experimental phase.

- **Ephemeral Storage**: Provides a simple, local ephemeral storage for pods.
- **Snapshot Support**: (Planned) Support for periodic snapshots and restoring from them.
