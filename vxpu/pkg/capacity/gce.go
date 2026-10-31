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

package capacity

import (
	"context"
	"fmt"
	"strings"

	compute "cloud.google.com/go/compute/apiv1"
	computepb "cloud.google.com/go/compute/apiv1/computepb"
	"google.golang.org/protobuf/proto"
)

// GCEBackend communicates with the Google Compute Engine API.
type GCEBackend struct {
	client  *compute.InstancesClient
	project string
}

// NewGCEBackend constructs a GCEBackend for the given project.
func NewGCEBackend(ctx context.Context, project string) (*GCEBackend, error) {
	if project == "" {
		return nil, fmt.Errorf("GCE project must not be empty")
	}
	client, err := compute.NewInstancesRESTClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create Compute Engine instances client: %w", err)
	}
	return &GCEBackend{
		client:  client,
		project: project,
	}, nil
}

// Close closes the underlying Compute Engine client connection.
func (b *GCEBackend) Close() error {
	if b.client != nil {
		return b.client.Close()
	}
	return nil
}

// Create requests a new flex-start instance in the specified zone.
func (b *GCEBackend) Create(ctx context.Context, zone string, spec MachineSpec) error {
	machineType := spec.MachineType
	if !strings.Contains(machineType, "/") {
		machineType = fmt.Sprintf("zones/%s/machineTypes/%s", zone, machineType)
	}

	image := spec.Image
	if image == "" {
		image = "projects/debian-cloud/global/images/family/debian-12"
	}

	diskType := spec.BootDiskType
	if diskType == "" {
		diskType = "hyperdisk-balanced"
	}
	if !strings.Contains(diskType, "/") {
		diskType = fmt.Sprintf("zones/%s/diskTypes/%s", zone, diskType)
	}

	diskSizeGB := spec.BootDiskSizeGB
	if diskSizeGB <= 0 {
		diskSizeGB = 100
	}

	disks := []*computepb.AttachedDisk{
		{
			Boot:       proto.Bool(true),
			AutoDelete: proto.Bool(true),
			Type:       proto.String("PERSISTENT"),
			InitializeParams: &computepb.AttachedDiskInitializeParams{
				SourceImage: proto.String(image),
				DiskType:    proto.String(diskType),
				DiskSizeGb:  proto.Int64(diskSizeGB),
			},
		},
	}

	for i := 0; i < spec.LocalSSDs; i++ {
		disks = append(disks, &computepb.AttachedDisk{
			Type:       proto.String("SCRATCH"),
			Interface:  proto.String("NVME"),
			AutoDelete: proto.Bool(true),
			InitializeParams: &computepb.AttachedDiskInitializeParams{
				DiskType: proto.String(fmt.Sprintf("zones/%s/diskTypes/local-ssd", zone)),
			},
		})
	}

	var guestAccelerators []*computepb.AcceleratorConfig
	for _, acc := range spec.Accelerators {
		accType := acc.Type
		if !strings.Contains(accType, "/") {
			accType = fmt.Sprintf("zones/%s/acceleratorTypes/%s", zone, accType)
		}
		guestAccelerators = append(guestAccelerators, &computepb.AcceleratorConfig{
			AcceleratorType:  proto.String(accType),
			AcceleratorCount: proto.Int32(int32(acc.Count)),
		})
	}

	network := spec.Network
	if network == "" {
		network = "global/networks/default"
	}

	networkInterfaces := []*computepb.NetworkInterface{
		{
			Network: proto.String(network),
			AccessConfigs: []*computepb.AccessConfig{
				{
					Type: proto.String("ONE_TO_ONE_NAT"),
					Name: proto.String("External NAT"),
				},
			},
		},
	}

	scheduling := &computepb.Scheduling{
		ProvisioningModel:         proto.String("FLEX_START"),
		OnHostMaintenance:         proto.String("TERMINATE"),
		InstanceTerminationAction: proto.String("DELETE"),
	}
	if spec.MaxRunDuration > 0 {
		scheduling.MaxRunDuration = &computepb.Duration{
			Seconds: proto.Int64(int64(spec.MaxRunDuration.Seconds())),
		}
	}

	var params *computepb.InstanceParams
	if spec.RequestValidDuration > 0 {
		params = &computepb.InstanceParams{
			RequestValidForDuration: &computepb.Duration{
				Seconds: proto.Int64(int64(spec.RequestValidDuration.Seconds())),
			},
		}
	}

	var serviceAccounts []*computepb.ServiceAccount
	if spec.ServiceAccount != "" || len(spec.Scopes) > 0 {
		sa := spec.ServiceAccount
		if sa == "" {
			sa = "default"
		}
		scopes := spec.Scopes
		if len(scopes) == 0 {
			scopes = []string{"https://www.googleapis.com/auth/cloud-platform"}
		}
		serviceAccounts = []*computepb.ServiceAccount{
			{
				Email:  proto.String(sa),
				Scopes: scopes,
			},
		}
	}

	inst := &computepb.Instance{
		Name:              proto.String(spec.Name),
		MachineType:       proto.String(machineType),
		Disks:             disks,
		GuestAccelerators: guestAccelerators,
		NetworkInterfaces: networkInterfaces,
		Scheduling:        scheduling,
		Params:            params,
		Labels:            spec.Labels,
		ServiceAccounts:   serviceAccounts,
	}

	req := &computepb.InsertInstanceRequest{
		Project:          b.project,
		Zone:             zone,
		InstanceResource: inst,
	}

	_, err := b.client.Insert(ctx, req)
	return err
}

// Describe returns the current status of the instance in the specified zone.
func (b *GCEBackend) Describe(ctx context.Context, zone string, name string) (*Instance, error) {
	req := &computepb.GetInstanceRequest{
		Project:  b.project,
		Zone:     zone,
		Instance: name,
	}
	inst, err := b.client.Get(ctx, req)
	if err != nil {
		if IsNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	return &Instance{
		Name:          name,
		Zone:          zone,
		Status:        InstanceStatus(inst.GetStatus()),
		StatusMessage: inst.GetStatusMessage(),
	}, nil
}

// Delete removes the instance in the specified zone.
func (b *GCEBackend) Delete(ctx context.Context, zone string, name string) error {
	req := &computepb.DeleteInstanceRequest{
		Project:  b.project,
		Zone:     zone,
		Instance: name,
	}
	_, err := b.client.Delete(ctx, req)
	if err != nil {
		if IsNotFound(err) {
			return ErrNotFound
		}
		return err
	}
	return nil
}
