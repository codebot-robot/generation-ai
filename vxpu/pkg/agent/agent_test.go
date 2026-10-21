// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightdisk"
)

// fakeDisk writes a weights disk's contents into a directory, as the
// factory would. The fake mount in server links to that directory.
func fakeDisk(t *testing.T, contents map[string]string) (string, map[string]int64) {
	t.Helper()
	disk := t.TempDir()
	if err := os.MkdirAll(filepath.Join(disk, weightdisk.BlobDir), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]int64{}
	for _, c := range contents {
		sum := sha256.Sum256([]byte(c))
		sha := hex.EncodeToString(sum[:])
		if err := os.WriteFile(filepath.Join(disk, weightdisk.BlobDir, sha), []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
		files[sha] = int64(len(c))
	}
	data, err := json.Marshal(weightdisk.Index{Format: weightdisk.Format, Source: "gs://bucket", Files: files})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(disk, weightdisk.IndexFile), data, 0o644); err != nil {
		t.Fatal(err)
	}
	return disk, files
}

func server(t *testing.T, disk string, mounts *[]string) *Server {
	return &Server{
		MountRoot: t.TempDir(),
		Mount: func(_ context.Context, device, dir string) error {
			*mounts = append(*mounts, device)
			if err := os.Remove(dir); err != nil {
				return err
			}
			return os.Symlink(disk, dir)
		},
		Unmount: func(_ context.Context, dir string) error { return os.Remove(dir) },
		// the fake mount replaces the directory with a symlink to the fake disk
		MountPoint: func(dir string) bool { fi, err := os.Lstat(dir); return err == nil && fi.Mode()&os.ModeSymlink != 0 },
	}
}

func TestMountWeights(t *testing.T) {
	ctx := t.Context()
	disk, files := fakeDisk(t, map[string]string{"a": "the first checkpoint file", "b": "and the second"})
	var mounts []string
	s := server(t, disk, &mounts)
	resp, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights", Verify: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(mounts) != 1 || mounts[0] != "/dev/disk/by-id/google-vxpu-weights" {
		t.Errorf("mounted %v, want the vxpu-weights device once", mounts)
	}
	if resp.GetBlobDir() != filepath.Join(s.MountRoot, "vxpu-weights", weightdisk.BlobDir) {
		t.Errorf("blob dir %q", resp.GetBlobDir())
	}
	if len(resp.GetFiles()) != len(files) || resp.GetTotalBytes() != int64(len("the first checkpoint file")+len("and the second")) {
		t.Errorf("files %v, %d bytes", resp.GetFiles(), resp.GetTotalBytes())
	}
	// mounting again is a no-op: the server doesn't mount a mounted disk twice
	if _, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}); err != nil || len(mounts) != 1 {
		t.Errorf("second mount: %v, mounts %v", err, mounts)
	}
	if _, err := s.UnmountWeights(ctx, &pb.UnmountWeightsRequest{DeviceName: "vxpu-weights"}); err != nil {
		t.Fatal(err)
	}
	if s.mounted(filepath.Join(s.MountRoot, "vxpu-weights")) {
		t.Error("still mounted after UnmountWeights")
	}
}

func TestMountWeightsCatchesACorruptFile(t *testing.T) {
	ctx := t.Context()
	disk, files := fakeDisk(t, map[string]string{"a": "the first checkpoint file"})
	for sha := range files { // same size, different bytes
		if err := os.WriteFile(filepath.Join(disk, weightdisk.BlobDir, sha), []byte("THE FIRST CHECKPOINT FILE"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var mounts []string
	s := server(t, disk, &mounts)
	if _, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}); err != nil {
		t.Fatalf("without verify, sizes are all that is checked: %v", err)
	}
	_, err := s.MountWeights(ctx, &pb.MountWeightsRequest{DeviceName: "vxpu-weights", Verify: true})
	if status.Code(err) != codes.DataLoss {
		t.Errorf("verify: %v, want DataLoss", err)
	}
}

func TestMountWeightsRefusesOddDeviceNames(t *testing.T) {
	s := &Server{MountRoot: t.TempDir()}
	for _, name := range []string{"", "../etc", "Weights", "a/b"} {
		_, err := s.MountWeights(t.Context(), &pb.MountWeightsRequest{DeviceName: name})
		if status.Code(err) != codes.InvalidArgument {
			t.Errorf("%q: %v, want InvalidArgument", name, err)
		}
	}
}

// The server unmounts a disk that has no weights index, rather than leave
// it where a weights disk belongs. A plain directory isn't a mount point.
func TestMountWeightsUnmountsANonWeightsDisk(t *testing.T) {
	var mounts []string
	s := server(t, t.TempDir(), &mounts) // an empty disk: no index
	if _, err := s.MountWeights(t.Context(), &pb.MountWeightsRequest{DeviceName: "vxpu-weights"}); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("err %v, want FailedPrecondition", err)
	}
	if len(mounts) != 1 || s.mounted(filepath.Join(s.MountRoot, "vxpu-weights")) {
		t.Errorf("mounts %v; still mounted after the index check failed", mounts)
	}
	if (&Server{}).mounted(t.TempDir()) {
		t.Error("a plain directory counts as a mount point")
	}
}
