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
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

type UploadOptions struct {
	ModelstoreURL string
	ModelName     string
	ModelDir      string
}

func (o *UploadOptions) InitDefaults() {
	o.ModelstoreURL = "http://localhost:8080"
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
	cmd.Flags().StringVar(&opt.ModelName, "model-name", "", "Name of the model")
	cmd.Flags().StringVar(&opt.ModelDir, "model-dir", "", "Directory containing the model files")

	cmd.MarkFlagRequired("model-name")
	cmd.MarkFlagRequired("model-dir")

	return cmd
}

func RunUpload(ctx context.Context, opt *UploadOptions) error {
	log := klog.FromContext(ctx)

	var files []v1alpha1.File

	err := filepath.Walk(opt.ModelDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		relPath, err := filepath.Rel(opt.ModelDir, path)
		if err != nil {
			return err
		}

		// Calculate SHA256
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		h := sha256.New()
		if _, err := io.Copy(h, f); err != nil {
			return err
		}
		sha := hex.EncodeToString(h.Sum(nil))

		files = append(files, v1alpha1.File{
			Path:   relPath,
			SHA256: sha,
		})

		// Check if blob already exists
		blobURL := fmt.Sprintf("%s/blobs/%s", strings.TrimSuffix(opt.ModelstoreURL, "/"), sha)
		resp, err := http.Head(blobURL)
		if err == nil && resp.StatusCode == http.StatusOK {
			log.V(2).Info("blob already exists", "sha", sha, "path", relPath)
			return nil
		}

		// Upload blob
		if _, err := f.Seek(0, 0); err != nil {
			return err
		}

		log.Info("uploading blob", "sha", sha, "path", relPath)
		req, err := http.NewRequestWithContext(ctx, http.MethodPut, blobURL, f)
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/octet-stream")

		resp, err = http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("failed to upload blob %s: %s %s", sha, resp.Status, string(body))
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk model directory: %w", err)
	}

	model := &v1alpha1.Model{
		ObjectMeta: metav1.ObjectMeta{
			Name: opt.ModelName,
		},
		Spec: v1alpha1.ModelSpec{
			Files: files,
		},
	}

	// Post completed model
	modelURL := fmt.Sprintf("%s/models", strings.TrimSuffix(opt.ModelstoreURL, "/"))
	body, err := json.Marshal(model)
	if err != nil {
		return err
	}

	log.Info("creating model", "name", opt.ModelName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, modelURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to create model %s: %s %s", opt.ModelName, resp.Status, string(body))
	}

	log.Info("successfully uploaded model", "name", opt.ModelName)
	return nil
}
