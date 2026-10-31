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
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"k8s.io/klog/v2"

	"github.com/gke-labs/generation-ai/vxpu/pkg/capacity"
)

// cmdRace runs the GPU capacity racer:
//
//	vxpu race --machine-type g4-standard-192 --zones us-central1-b,us-central1-c,us-east4-b [flags] [-- command...]
func cmdRace(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("race", flag.ContinueOnError)

	machineType := flags.String("machine-type", "", "GCE machine type (e.g. g4-standard-192, e2-standard-2)")
	zonesArg := flags.String("zones", "", "comma-separated list of zones to race in")
	project := flags.String("project", "", "GCP project ID (defaults to PROJECT or GOOGLE_CLOUD_PROJECT env, or metadata)")
	name := flags.String("name", "vxpu-race", "base name for the deterministic instances in each zone")
	image := flags.String("image", "projects/debian-cloud/global/images/family/debian-12", "boot disk image URL or family")
	bootDiskType := flags.String("boot-disk-type", "hyperdisk-balanced", "boot disk type (e.g. hyperdisk-balanced, pd-balanced)")
	bootDiskSizeGB := flags.Int64("boot-disk-size", 100, "boot disk size in GB")
	localSSDs := flags.Int("local-ssds", 0, "number of NVMe local SSDs")
	accelerator := flags.String("accelerator", "", "accelerator configuration (e.g. type=nvidia-rtx-pro-6000,count=4)")
	labels := flags.String("labels", "", "comma-separated key=value labels (e.g. purpose=vxpu-hero)")
	maxRunDuration := flags.Duration("max-run-duration", 7*24*time.Hour, "maximum duration the VM is allowed to run")
	requestValidity := flags.Duration("request-valid-for-duration", 2*time.Hour, "duration the flex-start request remains queued")
	timeout := flags.Duration("timeout", 2*time.Hour, "give up the race after this duration")
	pollInterval := flags.Duration("poll-interval", 2*time.Second, "polling interval for checking request status")
	var keepWinnerVal bool
	flags.BoolVar(&keepWinnerVal, "keep-winner", false, "keep the winner instance if interrupted or if bring-up fails")
	bringUpCmd := flags.String("bring-up", "", "optional command to run on the winner VM")
	jsonOutput := flags.Bool("json", false, "output winner details as JSON")

	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	if *machineType == "" || *zonesArg == "" {
		return errors.New("vxpu race requires --machine-type and --zones")
	}

	zones := parseZones(*zonesArg)
	if len(zones) == 0 {
		return errors.New("at least one valid zone must be provided in --zones")
	}

	proj := *project
	if proj == "" {
		proj = resolveDefaultProject(ctx)
		if proj == "" {
			return errors.New("GCP project not specified: use --project or set PROJECT environment variable")
		}
	}

	var accelerators []capacity.AcceleratorSpec
	if *accelerator != "" {
		acc, err := parseAccelerator(*accelerator)
		if err != nil {
			return err
		}
		accelerators = append(accelerators, acc)
	}

	parsedLabels := parseLabels(*labels)

	spec := capacity.MachineSpec{
		Name:                 *name,
		MachineType:          *machineType,
		Image:                *image,
		BootDiskType:         *bootDiskType,
		BootDiskSizeGB:       *bootDiskSizeGB,
		LocalSSDs:            *localSSDs,
		Accelerators:         accelerators,
		Labels:               parsedLabels,
		MaxRunDuration:       *maxRunDuration,
		RequestValidDuration: *requestValidity,
	}

	backend, err := capacity.NewGCEBackend(ctx, proj)
	if err != nil {
		return fmt.Errorf("connecting to GCE API: %w", err)
	}
	defer func() {
		if cerr := backend.Close(); cerr != nil {
			log.Error(cerr, "Closing GCE backend", "project", proj)
		}
	}()

	// Handle SIGINT and SIGTERM gracefully.
	sigCtx, stopSignals := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	raceOpts := capacity.RaceOptions{
		Spec:         spec,
		Zones:        zones,
		Timeout:      *timeout,
		PollInterval: *pollInterval,
		KeepWinner:   keepWinnerVal,
	}

	log.Info("Starting capacity race", "machine_type", *machineType, "zones", zones, "project", proj)
	winner, err := capacity.Race(sigCtx, backend, raceOpts)
	if err != nil {
		return fmt.Errorf("capacity race failed: %w", err)
	}

	// Print the winner to stdout.
	waitedDuration := winner.Wait.Round(time.Second)
	if *jsonOutput {
		out := map[string]any{
			"name":         winner.Name,
			"zone":         winner.Zone,
			"waited":       waitedDuration.String(),
			"wait_seconds": int64(waitedDuration.Seconds()),
		}
		data, err := json.Marshal(out)
		if err != nil {
			return err
		}
		fmt.Println(string(data))
	} else {
		fmt.Printf("%s %s %s\n", winner.Name, winner.Zone, waitedDuration)
	}

	// Run optional bring-up command if requested.
	cmdArgs := flags.Args()
	var execArgs []string
	if len(cmdArgs) > 0 {
		execArgs = cmdArgs
	} else if *bringUpCmd != "" {
		execArgs = []string{"sh", "-c", *bringUpCmd}
	}

	if len(execArgs) > 0 {
		log.Info("Running bring-up command on winner", "winner", winner.Name, "zone", winner.Zone, "command", execArgs)
		cmd := exec.CommandContext(sigCtx, execArgs[0], execArgs[1:]...)
		cmd.Env = append(os.Environ(),
			fmt.Sprintf("VM=%s", winner.Name),
			fmt.Sprintf("ZONE=%s", winner.Zone),
			fmt.Sprintf("NAME=%s", winner.Name),
			fmt.Sprintf("INSTANCE_NAME=%s", winner.Name),
			fmt.Sprintf("PROJECT=%s", proj),
		)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if cmdErr := cmd.Run(); cmdErr != nil {
			log.Error(cmdErr, "Bring-up command failed", "winner", winner.Name, "zone", winner.Zone)
			if !keepWinnerVal {
				log.Info("Deleting winner instance due to bring-up failure", "winner", winner.Name, "zone", winner.Zone)
				delCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Minute)
				defer cancel()
				if derr := backend.Delete(delCtx, winner.Zone, winner.Name); derr != nil && !capacity.IsNotFound(derr) {
					log.Error(derr, "Failed to delete winner instance", "winner", winner.Name, "zone", winner.Zone)
				}
			}
			return cmdErr
		}
	}

	return nil
}

func parseZones(s string) []string {
	var zones []string
	// Allow comma or space delimiters.
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == '\n'
	})
	for _, f := range fields {
		z := strings.TrimSpace(f)
		if z != "" {
			zones = append(zones, z)
		}
	}
	return zones
}

func parseLabels(s string) map[string]string {
	labels := make(map[string]string)
	if s == "" {
		return labels
	}
	parts := strings.Split(s, ",")
	for _, p := range parts {
		kv := strings.SplitN(strings.TrimSpace(p), "=", 2)
		if len(kv) == 2 {
			labels[kv[0]] = kv[1]
		}
	}
	return labels
}

func parseAccelerator(s string) (capacity.AcceleratorSpec, error) {
	if s == "" {
		return capacity.AcceleratorSpec{}, nil
	}
	if strings.Contains(s, "type=") {
		parts := strings.Split(s, ",")
		var accType string
		count := 1
		for _, p := range parts {
			kv := strings.SplitN(strings.TrimSpace(p), "=", 2)
			if len(kv) == 2 {
				switch kv[0] {
				case "type":
					accType = kv[1]
				case "count":
					c, err := strconv.Atoi(kv[1])
					if err != nil {
						return capacity.AcceleratorSpec{}, fmt.Errorf("invalid accelerator count %q: %w", kv[1], err)
					}
					count = c
				}
			}
		}
		if accType == "" {
			return capacity.AcceleratorSpec{}, fmt.Errorf("accelerator spec must include type: %q", s)
		}
		return capacity.AcceleratorSpec{Type: accType, Count: count}, nil
	}
	if strings.Contains(s, ":") {
		parts := strings.SplitN(s, ":", 2)
		count, err := strconv.Atoi(parts[1])
		if err != nil {
			return capacity.AcceleratorSpec{}, fmt.Errorf("invalid accelerator count in %q: %w", s, err)
		}
		return capacity.AcceleratorSpec{Type: parts[0], Count: count}, nil
	}
	return capacity.AcceleratorSpec{Type: s, Count: 1}, nil
}

func resolveDefaultProject(ctx context.Context) string {
	log := klog.FromContext(ctx)
	if p := os.Getenv("PROJECT"); p != "" {
		return p
	}
	if p := os.Getenv("GOOGLE_CLOUD_PROJECT"); p != "" {
		return p
	}
	reqCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "GET", "http://metadata.google.internal/computeMetadata/v1/project/project-id", nil)
	if err != nil {
		log.V(2).Info("Failed to create metadata request", "error", err)
		return ""
	}
	req.Header.Set("Metadata-Flavor", "Google")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.V(2).Info("Failed to query metadata server", "error", err)
		return ""
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			log.V(2).Info("Failed to close metadata response body", "error", cerr)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil || len(data) == 0 {
		return ""
	}
	return strings.TrimSpace(string(data))
}
