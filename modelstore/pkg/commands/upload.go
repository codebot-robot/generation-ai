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

package commands

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gke-labs/generation-ai/modelstore/pkg/registry"
	"github.com/spf13/cobra"
	"k8s.io/klog/v2"
)

type UploadOptions struct {
	ModelstoreURL string
	ModelName     string
	ModelDir      string
	Tag           string
}

func (o *UploadOptions) InitDefaults() {
	o.ModelstoreURL = "http://localhost:8080"
	o.Tag = "latest"
}

func BuildUploadCommand() *cobra.Command {
	opt := &UploadOptions{}
	opt.InitDefaults()

	cmd := &cobra.Command{
		Use:   "upload",
		Short: "Upload a model to the modelstore",
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunUpload(cmd.Context(), opt)
		},
	}

	cmd.Flags().StringVar(&opt.ModelstoreURL, "modelstore-url", opt.ModelstoreURL, "URL of the modelstore server")
	cmd.Flags().StringVar(&opt.ModelName, "model-name", "", "Registry name for the model (lowercase, e.g. org/name)")
	cmd.Flags().StringVar(&opt.ModelDir, "model-dir", "", "Directory containing the model files")
	cmd.Flags().StringVar(&opt.Tag, "tag", opt.Tag, "Tag to write")

	if err := cmd.MarkFlagRequired("model-name"); err != nil {
		panic(err)
	}
	if err := cmd.MarkFlagRequired("model-dir"); err != nil {
		panic(err)
	}

	return cmd
}

// RunUpload pushes a directory as an OCI artifact: every file as a
// blob (skipped when the registry already has its digest), then a
// manifest listing them by name, tagged <model-name>:<tag>.
func RunUpload(ctx context.Context, opt *UploadOptions) error {
	log := klog.FromContext(ctx)
	name := opt.ModelName
	if !registry.ValidName(name) {
		return fmt.Errorf("model name %q is not a valid registry name (lowercase, [a-z0-9._-/])", name)
	}
	base := strings.TrimSuffix(opt.ModelstoreURL, "/")
	client := &http.Client{}

	var layers []registry.Descriptor
	err := filepath.Walk(opt.ModelDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		// Walk reports symlinks as themselves; a Hub cache snapshot is a
		// tree of symlinks into blobs/, so size (and dir-ness) come from
		// the target, which is also what fileSHA256 hashes.
		if info, err = os.Stat(path); err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		relPath, err := filepath.Rel(opt.ModelDir, path)
		if err != nil {
			return err
		}
		relPath = filepath.ToSlash(relPath)
		sha, err := fileSHA256(path)
		if err != nil {
			return err
		}
		digest := "sha256:" + sha
		layers = append(layers, registry.Descriptor{MediaType: registry.MediaTypeFile, Digest: digest, Size: info.Size(),
			Annotations: map[string]string{registry.AnnotationTitle: relPath}})

		head, err := http.NewRequestWithContext(ctx, http.MethodHead, base+"/v2/"+name+"/blobs/"+digest, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(head)
		if err != nil {
			return fmt.Errorf("checking blob %s: %w", digest, err)
		}
		closeBody(ctx, resp)
		if resp.StatusCode == http.StatusOK {
			log.V(2).Info("blob already present", "digest", digest, "path", relPath)
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer func() {
			if err := f.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
				log.Error(err, "closing file", "path", path)
			}
		}()
		log.Info("uploading blob", "digest", digest, "path", relPath, "bytes", info.Size())
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v2/"+name+"/blobs/uploads/?digest="+digest, f)
		if err != nil {
			return err
		}
		req.ContentLength = info.Size()
		req.Header.Set("Content-Type", "application/octet-stream")
		return expect(client, req, http.StatusCreated)
	})
	if err != nil {
		return fmt.Errorf("uploading %s: %w", opt.ModelDir, err)
	}

	manifest := registry.Manifest{SchemaVersion: 2, MediaType: registry.MediaTypeManifest,
		ArtifactType: registry.ArtifactWeights, Config: registry.EmptyConfig, Layers: layers}
	body, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	log.Info("writing manifest", "name", name, "tag", opt.Tag, "files", len(layers))
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, base+"/v2/"+name+"/manifests/"+opt.Tag, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", registry.MediaTypeManifest)
	if err := expect(client, req, http.StatusCreated); err != nil {
		return fmt.Errorf("writing manifest: %w", err)
	}
	log.Info("uploaded model", "reference", name+":"+opt.Tag)
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	_, err = io.Copy(h, f)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func closeBody(ctx context.Context, resp *http.Response) {
	if err := resp.Body.Close(); err != nil {
		klog.FromContext(ctx).Error(err, "closing response body", "url", resp.Request.URL.String())
	}
}

// expect runs req and fails unless the status is want.
func expect(client *http.Client, req *http.Request, want int) error {
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer closeBody(req.Context(), resp)
	if resp.StatusCode != want {
		msg, rerr := io.ReadAll(io.LimitReader(resp.Body, 2048))
		if rerr != nil {
			msg = []byte("(unreadable body)")
		}
		return fmt.Errorf("%s %s: %s: %s", req.Method, req.URL.Path, resp.Status, strings.TrimSpace(string(msg)))
	}
	return nil
}
