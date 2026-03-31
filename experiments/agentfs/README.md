# AgentFS - Ephemeral CSI Driver

AgentFS is an experimental CSI driver that provides a simple ephemeral filesystem for agents.

## Vision

The goal of AgentFS is to provide a filesystem that is ephemeral by default but can be periodically snapshotted. This allows agents to have a persistent state at specific points in time without the overhead of a fully distributed POSIX filesystem.

## Status

AgentFS now supports basic snapshotting of ephemeral storage.

- **Ephemeral Storage**: Provides a simple, local ephemeral storage for pods.
- **Snapshot Support**: Implemented snapshotting (push/pull) to the AgentFS Controller.
- **AgentFS Controller**: A StatefulSet that manages snapshots and blobs, backed by a persistent volume.

## Snapshotting Mechanism

AgentFS implements snapshotting by calculating SHA256 checksums for each file in the volume. When a volume is unmounted, the node daemon:
1. Calculates checksums for all files.
2. Identifies new blobs (files) that are not already present in the AgentFS Controller.
3. Uploads new blobs and a metadata file containing file modes, timestamps, and checksums.

When a volume is mounted:
1. The node daemon pulls the latest snapshot metadata from the Controller.
2. It downloads missing blobs from the Controller.
3. It populates the local ephemeral storage and restores file attributes (modes, timestamps).

This allows agents to persist state across different pod instances and nodes while maintaining the performance of local ephemeral storage.
