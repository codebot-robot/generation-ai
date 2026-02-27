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

package main

import (
	"context"
	"os"

	"github.com/gke-labs/generation-ai/modelstore/pkg/commands"
	"github.com/spf13/cobra"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	defer klog.Flush()

	ctx := context.Background()

	rootCmd := &cobra.Command{
		Use:   "modelstore",
		Short: "Modelstore management CLI",
	}

	rootCmd.AddCommand(commands.BuildServeCommand())
	rootCmd.AddCommand(commands.BuildUploadCommand())

	if err := rootCmd.ExecuteContext(ctx); err != nil {
		klog.ErrorS(err, "terminated with error")
		os.Exit(1)
	}
}
