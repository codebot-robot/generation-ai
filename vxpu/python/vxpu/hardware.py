# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware model schema and machine catalog for the vXPU planner cost model.

Provides Python dataclasses (JSON-serialisable) describing machine resources,
capacities, speeds, prices, and telemetry-calibrated measurements, alongside
a catalog of reference hardware and a probe() routine for the current machine.
"""

import argparse
import dataclasses
import json
import sys
from typing import Any, Optional

from .plan import probe_hardware


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ComputeByDtype:
    """Peak compute throughput in FLOPS (floating-point operations per second)."""
    bf16: float  # BFLOAT16 dense peak FLOPS
    fp8: float   # FP8 dense peak FLOPS (0.0 if unsupported)
    fp32: float  # FP32 peak FLOPS


@dataclasses.dataclass(frozen=True)
class GPU:
    """Specification of an accelerator GPU (per device)."""
    name: str                                             # GPU device model name
    memory_bytes: int                                     # VRAM capacity in bytes
    memory_bandwidth_bytes_per_sec: Optional[float] = None  # VRAM memory bandwidth in bytes/sec
    peak_compute_flops: Optional[ComputeByDtype] = None   # Peak compute throughput by precision
    sm_count: Optional[int] = None                        # Streaming Multiprocessor count (or GPU cores)
    pcie_gen: Optional[int] = None                        # PCIe generation (e.g. 4, 5) or None if unified/fabric
    pcie_lanes: Optional[int] = None                      # PCIe lane count (e.g. 16) or None
    pcie_bandwidth_bytes_per_sec: Optional[float] = None  # Host<->device bandwidth each way (bytes/sec)


@dataclasses.dataclass(frozen=True)
class CPU:
    """Specification of the host CPU."""
    model: str                                            # CPU model string
    cores: int                                            # Physical CPU cores
    threads: int                                          # Logical threads / vCPUs
    simd: str                                             # SIMD capability level: "avx512", "avx2", "neon", etc.
    memory_bandwidth_bytes_per_sec: Optional[float] = None  # Aggregate system memory bandwidth in bytes/sec
    cache_sizes: Optional[dict[str, int]] = None          # Cache level sizes in bytes, e.g. {"l1d": ..., "l2": ..., "l3": ...}


@dataclasses.dataclass(frozen=True)
class LocalStorage:
    """Specification of local high-speed storage (e.g. NVMe Local SSD array)."""
    devices: int                                          # Number of attached NVMe devices
    capacity_bytes: int                                   # Total capacity in bytes
    read_bandwidth_bytes_per_sec: float                   # Aggregate sequential read bandwidth (bytes/sec)
    write_bandwidth_bytes_per_sec: float                  # Aggregate sequential write bandwidth (bytes/sec)
    read_iops: float                                      # Aggregate random 4K read IOPS
    write_iops: float                                     # Aggregate random 4K write IOPS


@dataclasses.dataclass(frozen=True)
class Network:
    """Specification of network connectivity."""
    bandwidth_bytes_per_sec: float                        # Peak network bandwidth in bytes/sec


@dataclasses.dataclass(frozen=True)
class Price:
    """Cloud rental pricing in USD per hour."""
    on_demand_usd_per_hour: float                         # On-demand price per hour
    spot_usd_per_hour: float                              # Spot / preemptible price per hour


@dataclasses.dataclass(frozen=True)
class Machine:
    """Hardware model for a physical or cloud machine."""
    name: str                                             # Machine identifier (e.g. "g4-standard-96")
    cpu: CPU                                              # CPU specification
    ram_bytes: int                                        # System RAM capacity in bytes
    gpu: Optional[GPU] = None                             # GPU specification (per device) if present
    gpu_count: int = 0                                    # Number of GPUs attached
    storage: Optional[LocalStorage] = None                # Local NVMe storage if present
    network: Optional[Network] = None                     # Network interface specification
    price: Optional[Price] = None                         # Hourly price specification
    measured: Optional[dict[str, Any]] = None             # Benchmark measurements and telemetry
    description: str = ""                                 # Human-readable description

    def __post_init__(self):
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert machine model to a plain JSON-serialisable dictionary."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Format machine model as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """A formatted, human-readable summary of machine specifications."""
        lines = [f"Machine: {self.name}"]
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.append(
            f"  CPU: {self.cpu.model}, {self.cpu.cores} cores / {self.cpu.threads} threads, SIMD: {self.cpu.simd}"
        )
        if self.cpu.memory_bandwidth_bytes_per_sec is not None:
            lines.append(f"    Memory Bandwidth: {self.cpu.memory_bandwidth_bytes_per_sec / 1e9:.1f} GB/s")
        if self.cpu.cache_sizes:
            cache_str = ", ".join(f"{k.upper()}: {v / (1024 * 1024):.1f} MB" for k, v in sorted(self.cpu.cache_sizes.items()))
            lines.append(f"    Cache: {cache_str}")
        lines.append(f"  RAM: {self.ram_bytes / (1024**3):.1f} GB ({self.ram_bytes:,} bytes)")
        if self.gpu and self.gpu_count > 0:
            lines.append(f"  GPU: {self.gpu_count}x {self.gpu.name}")
            lines.append(f"    VRAM: {self.gpu.memory_bytes / (1024**3):.1f} GB per device ({self.gpu.memory_bytes:,} bytes)")
            if self.gpu.memory_bandwidth_bytes_per_sec is not None:
                lines.append(f"    Memory Bandwidth: {self.gpu.memory_bandwidth_bytes_per_sec / 1e9:.1f} GB/s")
            if self.gpu.peak_compute_flops is not None:
                p = self.gpu.peak_compute_flops
                lines.append(f"    Peak Compute: BF16: {p.bf16 / 1e12:.1f} TFLOPS, FP8: {p.fp8 / 1e12:.1f} TFLOPS, FP32: {p.fp32 / 1e12:.1f} TFLOPS")
            if self.gpu.sm_count is not None:
                lines.append(f"    SMs: {self.gpu.sm_count}")
            if self.gpu.pcie_gen:
                pcie_bw = f"{self.gpu.pcie_bandwidth_bytes_per_sec / 1e9:.1f} GB/s" if self.gpu.pcie_bandwidth_bytes_per_sec else "n/a"
                lines.append(f"    PCIe: Gen {self.gpu.pcie_gen} x{self.gpu.pcie_lanes} ({pcie_bw} each way)")
        else:
            lines.append("  GPU: None")
        if self.storage:
            lines.append(
                f"  Storage: {self.storage.devices} devices, {self.storage.capacity_bytes / 1e9:.0f} GB Local SSD"
            )
            lines.append(
                f"    Read: {self.storage.read_bandwidth_bytes_per_sec / 1e9:.2f} GB/s, Write: {self.storage.write_bandwidth_bytes_per_sec / 1e9:.2f} GB/s, "
                f"IOPS: {self.storage.read_iops:,.0f} read / {self.storage.write_iops:,.0f} write"
            )
        if self.network:
            lines.append(f"  Network: {self.network.bandwidth_bytes_per_sec * 8 / 1e9:.0f} Gbps ({self.network.bandwidth_bytes_per_sec / 1e9:.2f} GB/s)")
        if self.price:
            lines.append(f"  Price: ${self.price.on_demand_usd_per_hour:.2f}/hr on-demand, ${self.price.spot_usd_per_hour:.2f}/hr Spot")
        if self.measured:
            lines.append("  Measured:")
            for k, v in sorted(self.measured.items()):
                if isinstance(v, (int, float)):
                    if "bytes_per_sec" in k or "bytes_per_s" in k:
                        lines.append(f"    {k}: {v / 1e9:.2f} GB/s")
                    elif "bytes" in k:
                        lines.append(f"    {k}: {v / (1024**2):.1f} MiB ({v:,} bytes)")
                    elif "iops" in k:
                        lines.append(f"    {k}: {v:,.0f} IOPS")
                    elif k.endswith("_ms") or "time_ms" in k:
                        lines.append(f"    {k}: {v:.1f} ms")
                    else:
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def validate(self) -> bool:
        """Validate that all entries have positive numbers and consistent units."""
        if not self.name:
            raise ValueError("Machine name must be non-empty string")
        if not self.cpu or not self.cpu.model:
            raise ValueError(f"{self.name}: cpu and cpu.model must be set")
        if self.cpu.cores <= 0 or self.cpu.threads < self.cpu.cores:
            raise ValueError(f"{self.name}: invalid CPU core/thread counts: {self.cpu.cores}/{self.cpu.threads}")
        if self.gpu is not None:
            if not self.gpu.name:
                raise ValueError(f"{self.name}: gpu.name must be non-empty string")
            if self.gpu_count <= 0:
                raise ValueError(f"{self.name}: gpu_count must be positive when gpu is present")
        elif self.gpu_count != 0:
            raise ValueError(f"{self.name}: gpu_count must be 0 when gpu is None")

        checks = [
            ("ram_bytes", self.ram_bytes, 1 << 20, "bytes"),
            ("cpu.memory_bandwidth_bytes_per_sec", self.cpu.memory_bandwidth_bytes_per_sec, 1e9, "B/s"),
            ("gpu.memory_bytes", self.gpu and self.gpu.memory_bytes, 1 << 30, "bytes"),
            ("gpu.memory_bandwidth_bytes_per_sec", self.gpu and self.gpu.memory_bandwidth_bytes_per_sec, 1e9, "B/s"),
            ("gpu.peak_compute_flops.fp32", self.gpu and self.gpu.peak_compute_flops and self.gpu.peak_compute_flops.fp32, 1e9, "FLOPS"),
            ("gpu.peak_compute_flops.bf16", self.gpu and self.gpu.peak_compute_flops and self.gpu.peak_compute_flops.bf16, 0.0, "FLOPS"),
            ("gpu.peak_compute_flops.fp8", self.gpu and self.gpu.peak_compute_flops and self.gpu.peak_compute_flops.fp8, 0.0, "FLOPS"),
            ("gpu.sm_count", self.gpu and self.gpu.sm_count, 0, "SM count"),
            ("gpu.pcie_gen", self.gpu and self.gpu.pcie_gen, 1, "PCIe gen"),
            ("gpu.pcie_lanes", self.gpu and self.gpu.pcie_lanes, 1, "PCIe lanes"),
            ("gpu.pcie_bandwidth_bytes_per_sec", self.gpu and self.gpu.pcie_bandwidth_bytes_per_sec, 1e9, "B/s"),
            ("storage.devices", self.storage and self.storage.devices, 1, "devices"),
            ("storage.capacity_bytes", self.storage and self.storage.capacity_bytes, 1e9, "bytes"),
            ("storage.read_bandwidth_bytes_per_sec", self.storage and self.storage.read_bandwidth_bytes_per_sec, 1e6, "B/s"),
            ("storage.write_bandwidth_bytes_per_sec", self.storage and self.storage.write_bandwidth_bytes_per_sec, 1e6, "B/s"),
            ("storage.read_iops", self.storage and self.storage.read_iops, 1000, "IOPS"),
            ("storage.write_iops", self.storage and self.storage.write_iops, 1000, "IOPS"),
            ("network.bandwidth_bytes_per_sec", self.network and self.network.bandwidth_bytes_per_sec, 1e6, "B/s"),
            ("price.on_demand_usd_per_hour", self.price and self.price.on_demand_usd_per_hour, 0.0, "USD/hr"),
            ("price.spot_usd_per_hour", self.price and self.price.spot_usd_per_hour, 0.0, "USD/hr"),
        ]
        for field, value, minimum, unit in checks:
            if value is not None and value < minimum:
                raise ValueError(f"{self.name}: {field} {value} is too small; expected {unit}")

        if self.cpu.cache_sizes:
            for k, v in self.cpu.cache_sizes.items():
                if v <= 0:
                    raise ValueError(f"{self.name}: cpu.cache_sizes[{k}] {v} is too small; expected bytes")

        if self.measured:
            for k, v in self.measured.items():
                if isinstance(v, (int, float)) and v <= 0:
                    raise ValueError(f"{self.name}: measured[{k}] {v} must be positive")

        return True


# ---------------------------------------------------------------------------
# Machine Catalog
# ---------------------------------------------------------------------------

G4_STANDARD_48 = Machine(
    name="g4-standard-48",
    description="GCP G4 instance with 1x NVIDIA RTX PRO 6000 Blackwell Server Edition GPU",
    cpu=CPU(
        model="AMD EPYC 9B45 (Turin)",  # GCP G4 machine family spec sheet (AMD 5th Gen Turin custom CPU)
        cores=24,  # 24 physical cores: GCP G4 family spec (48 vCPUs / 2 threads per core)
        threads=48,  # 48 vCPUs: GCP G4 family spec sheet (g4-standard-48)
        simd="avx512",  # AMD Zen 5 (Turin) architecture whitepaper: full 512-bit AVX-512
        memory_bandwidth_bytes_per_sec=144.0e9,  # 144.0 GB/s (estimate: 48/192 proportional share of 576.0 GB/s Turin socket peak: 12 channels DDR5-6000, 12 * 8 * 6.0 GB/s)
        cache_sizes={
            "l1d": 24 * 48 * 1024,  # 48 KB L1 data per core x 24 cores = 1.15 MB: AMD Zen 5 spec
            "l1i": 24 * 32 * 1024,  # 32 KB L1 instruction per core x 24 cores = 0.77 MB: AMD Zen 5 spec
            "l2": 24 * 1024 * 1024,  # 1 MB L2 per core x 24 cores = 24 MB: AMD Zen 5 spec
            "l3": 96 * 1024 * 1024,  # 32 MB L3 per CCD x 3 CCDs = 96 MB: AMD Zen 5 spec
        },
    ),
    ram_bytes=180 * (1024**3),  # 180 GiB (193,273,528,320 bytes): GCP G4 family spec sheet (g4-standard-48)
    gpu=GPU(
        name="NVIDIA RTX PRO 6000",  # NVIDIA Blackwell Server Edition (GCP G4 accelerator)
        memory_bytes=96 * (1024**3),  # 96 GiB (103,079,215,104 bytes) GDDR7: NVIDIA RTX PRO 6000 datasheet
        memory_bandwidth_bytes_per_sec=1_597.0e9,  # ~1.6 TB/s (1597 GB/s): NVIDIA RTX PRO 6000 Blackwell Server Edition datasheet (512-bit GDDR7 at 25 Gbps; Workstation is 1792 GB/s at 28 Gbps)
        peak_compute_flops=ComputeByDtype(
            bf16=500.0e12,  # ~500 TFLOPS dense (estimate: ~half of FP8 tensor-core throughput, NVIDIA Blackwell architecture whitepaper)
            fp8=1_008.0e12,  # 1,008 TFLOPS dense (estimate: 5th gen tensor cores, NVIDIA Blackwell architecture whitepaper)
            fp32=125.0e12,  # 125 TFLOPS (estimate: ~24,064 CUDA cores at ~2.6 GHz, NVIDIA Blackwell architecture whitepaper)
        ),
        sm_count=188,  # 188 SMs (estimate: GB202 server edition configuration, NVIDIA Blackwell architecture whitepaper)
        pcie_gen=5,  # PCIe Gen 5: NVIDIA RTX PRO 6000 datasheet
        pcie_lanes=16,  # 16 lanes: NVIDIA RTX PRO 6000 datasheet
        pcie_bandwidth_bytes_per_sec=63.0e9,  # ~63.0 GB/s each way: PCI-SIG PCIe 5.0 x16 theoretical bandwidth
    ),
    gpu_count=1,  # 1 GPU: GCP G4 family spec sheet (g4-standard-48)
    storage=LocalStorage(
        devices=4,  # 4 Local Titanium NVMe SSDs: GCP G4 attached local SSDs
        capacity_bytes=1_500 * (10**9),  # 1,500 GB (4 x 375 GB): GCP Local SSD documentation
        read_bandwidth_bytes_per_sec=2_621_440_000.0,  # 2,500 MiB/s (2.62 GB/s): https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-48)
        write_bandwidth_bytes_per_sec=1_384_120_320.0,  # 1,320 MiB/s (1.38 GB/s): https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-48)
        read_iops=600_000.0,  # 600,000 IOPS: https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-48)
        write_iops=300_000.0,  # 300,000 IOPS: https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-48)
    ),
    network=Network(
        bandwidth_bytes_per_sec=6.25e9,  # 50 Gbps (6.25 GB/s): GCP G4 networking specification (g4-standard-48)
    ),
    price=Price(
        on_demand_usd_per_hour=4.50,  # $4.50/hour on-demand: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, G4 family)
        spot_usd_per_hour=1.7149,  # $1.7149/hour Spot: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, G4 family)
    ),
)


G4_STANDARD_96 = Machine(
    name="g4-standard-96",
    description="GCP G4 instance with 2x NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (the dev VM)",
    cpu=CPU(
        model="AMD EPYC 9B45 (Turin)",  # GCP G4 machine family spec sheet (AMD 5th Gen Turin custom CPU)
        cores=48,  # 48 physical cores: GCP G4 family spec (96 vCPUs / 2 threads per core)
        threads=96,  # 96 vCPUs: GCP G4 family spec sheet (g4-standard-96)
        simd="avx512",  # AMD Zen 5 (Turin) architecture whitepaper: full 512-bit AVX-512
        memory_bandwidth_bytes_per_sec=288.0e9,  # 288.0 GB/s (estimate: 96/192 proportional share of 576.0 GB/s Turin socket peak: 12 channels DDR5-6000, 12 * 8 * 6.0 GB/s)
        cache_sizes={
            "l1d": 48 * 48 * 1024,  # 48 KB L1 data per core x 48 cores = 2.3 MB: AMD Zen 5 spec
            "l1i": 48 * 32 * 1024,  # 32 KB L1 instruction per core x 48 cores = 1.5 MB: AMD Zen 5 spec
            "l2": 48 * 1024 * 1024,  # 1 MB L2 per core x 48 cores = 48 MB: AMD Zen 5 spec
            "l3": 192 * 1024 * 1024,  # 32 MB L3 per CCD x 6 CCDs = 192 MB: AMD Zen 5 spec
        },
    ),
    ram_bytes=360 * (1024**3),  # 360 GiB (386,547,056,640 bytes): GCP G4 family spec sheet (g4-standard-96)
    gpu=GPU(
        name="NVIDIA RTX PRO 6000",  # NVIDIA Blackwell Server Edition (GCP G4 accelerator)
        memory_bytes=96 * (1024**3),  # 96 GiB (103,079,215,104 bytes) GDDR7: NVIDIA RTX PRO 6000 datasheet
        memory_bandwidth_bytes_per_sec=1_597.0e9,  # ~1.6 TB/s (1597 GB/s): NVIDIA RTX PRO 6000 Blackwell Server Edition datasheet (512-bit GDDR7 at 25 Gbps; Workstation is 1792 GB/s at 28 Gbps)
        peak_compute_flops=ComputeByDtype(
            bf16=500.0e12,  # ~500 TFLOPS dense (estimate: ~half of FP8 tensor-core throughput, NVIDIA Blackwell architecture whitepaper)
            fp8=1_008.0e12,  # 1,008 TFLOPS dense (estimate: 5th gen tensor cores, NVIDIA Blackwell architecture whitepaper)
            fp32=125.0e12,  # 125 TFLOPS (estimate: ~24,064 CUDA cores at ~2.6 GHz, NVIDIA Blackwell architecture whitepaper)
        ),
        sm_count=188,  # 188 SMs (estimate: GB202 server edition configuration, NVIDIA Blackwell architecture whitepaper)
        pcie_gen=5,  # PCIe Gen 5: NVIDIA RTX PRO 6000 datasheet
        pcie_lanes=16,  # 16 lanes: NVIDIA RTX PRO 6000 datasheet
        pcie_bandwidth_bytes_per_sec=63.0e9,  # ~63.0 GB/s each way: PCI-SIG PCIe 5.0 x16 theoretical bandwidth
    ),
    gpu_count=2,  # 2 GPUs: GCP G4 family spec sheet (g4-standard-96)
    storage=LocalStorage(
        devices=8,  # 8 Local Titanium NVMe SSDs: vxpu/dev/vm/create.sh and GCP G4 maximum attached local SSDs
        capacity_bytes=3_000 * (10**9),  # 3,000 GB (8 x 375 GB): GCP Local SSD documentation
        read_bandwidth_bytes_per_sec=5_242_880_000.0,  # 5,000 MiB/s (5.24 GB/s): https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-96)
        write_bandwidth_bytes_per_sec=2_768_240_640.0,  # 2,640 MiB/s (2.77 GB/s): https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-96)
        read_iops=1_200_000.0,  # 1,200,000 IOPS: https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-96)
        write_iops=600_000.0,  # 600,000 IOPS: https://cloud.google.com/compute/docs/disks/local-ssd (Titanium SSD performance table, g4-standard-96)
    ),
    network=Network(
        bandwidth_bytes_per_sec=12.5e9,  # 100 Gbps (12.5 GB/s): GCP G4 networking specification (g4-standard-96)
    ),
    price=Price(
        on_demand_usd_per_hour=8.9999,  # $8.9999/hour on-demand: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, G4 family)
        spot_usd_per_hour=3.4298,  # $3.4298/hour Spot: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, G4 family)
    ),
    measured={
        "gcs_blob_fetch_bytes_per_sec": 1.7e9,  # ~1.7 GB/s: 510 GB checkpoint fetched from GCS bucket via gcloud parallel client (vxpu/dev/vm/README.md)
        "gcs_checkpoint_copy_bytes_per_sec": 2.89e9,  # ~2.89 GB/s: 510.3 GB checkpoint copied from GCS bucket to local SSD in 176.7 s by vxpu-hero (local weight tier, #361)
        "checkpoint_load_bytes_per_sec": 2.83e9,  # ~2.83 GB/s: engine checkpoint load from 8-disk RAID0 into RAM (vxpu/dev/vm/README.md)
        "local_ssd_read_bytes_per_sec": 5.44e9,  # 5.44 GB/s (5,190 MiB/s): measured with fio on dev VM (8 Titanium local SSDs, RAID0, direct I/O)
        "local_ssd_read_iops": 1.33e6,  # 1.33M IOPS: measured with fio on dev VM (4 KiB random read at high queue depth; 74 us latency per read at QD 1)
        "cpu_fp4_expert_kernel_effective_bytes_per_sec": 2.09e11,  # ~209 GB/s effective: 240 expert pairs x 18.8 MB in 21.6 ms per token on 48 threads, all experts on CPU (#368 baseline)
        "usable_gpu_memory_bytes": 97887 * 1024 * 1024,  # 97,887 MiB (102.6 GB) per RTX PRO 6000 Server Edition via nvidia-smi vs 96 GiB nominal
    },
)


C3D_STANDARD_60 = Machine(
    name="c3d-standard-60",
    description="GCP C3D compute instance with AMD EPYC Genoa processors (60 vCPUs)",
    cpu=CPU(
        model="AMD EPYC 9B14",  # GCP C3D machine family spec sheet (AMD 4th Gen Genoa custom CPU)
        cores=30,  # 30 physical cores: GCP C3D family spec (60 vCPUs / 2 threads per core)
        threads=60,  # 60 vCPUs: GCP C3D family spec sheet (c3d-standard-60)
        simd="avx512",  # AMD Zen 4 (Genoa) architecture whitepaper: native AVX-512
        memory_bandwidth_bytes_per_sec=153.6e9,  # 153.6 GB/s (estimate: 60/180 proportional share of single-socket AMD EPYC 9004 460.8 GB/s peak: 12 channels DDR5-4800, 12 * 8 * 4.8 GB/s)
        cache_sizes={
            "l1d": 30 * 32 * 1024,  # 32 KB L1 data per core x 30 cores = 0.96 MB: AMD Zen 4 spec
            "l1i": 30 * 32 * 1024,  # 32 KB L1 instruction per core x 30 cores = 0.96 MB: AMD Zen 4 spec
            "l2": 30 * 1024 * 1024,  # 1 MB L2 per core x 30 cores = 30 MB: AMD Zen 4 spec
            "l3": 128 * 1024 * 1024,  # 128 MB L3: AMD Zen 4 spec (proportional share of 384 MB L3)
        },
    ),
    ram_bytes=240 * (1024**3),  # 240 GiB (257,698,037,760 bytes): GCP C3D spec sheet (c3d-standard-60)
    gpu=None,
    gpu_count=0,
    network=Network(
        bandwidth_bytes_per_sec=5.0e9,  # 40 Gbps (5.0 GB/s) default egress: GCP C3D networking documentation
    ),
    price=Price(
        on_demand_usd_per_hour=2.7239,  # $2.7239/hour on-demand: https://cloud.google.com/compute/all-pricing (us-central1, C3D)
        spot_usd_per_hour=0.6594,  # $0.6594/hour Spot: https://cloud.google.com/compute/all-pricing (us-central1, C3D)
    ),
    measured={
        "fp4_expert_decode_kernel_bytes_per_sec": 121.0e9,  # ~121 GB/s packed fp4 on 30 threads (commit 5b99d74 and issue #269 microbenchmarks)
        "gcs_blob_fetch_bytes_per_sec": 1.4e9,  # ~1.4 GB/s directly from GCS with 16 in flight on c3d pod (issue #275)
    },
)


C3D_STANDARD_180 = Machine(
    name="c3d-standard-180",
    description="GCP C3D compute instance with AMD EPYC Genoa processors (180 vCPUs)",
    cpu=CPU(
        model="AMD EPYC 9B14",  # GCP C3D machine family spec sheet (AMD 4th Gen Genoa custom CPU)
        cores=90,  # 90 physical cores: GCP C3D family spec (180 vCPUs / 2 threads per core)
        threads=180,  # 180 vCPUs: GCP C3D family spec sheet (c3d-standard-180)
        simd="avx512",  # AMD Zen 4 (Genoa) architecture whitepaper: native AVX-512
        memory_bandwidth_bytes_per_sec=460.8e9,  # 460.8 GB/s (estimate: full single-socket AMD EPYC 9004 peak: 12 channels DDR5-4800, 12 * 8 * 4.8 GB/s)
        cache_sizes={
            "l1d": 90 * 32 * 1024,  # 32 KB L1 data per core x 90 cores = 2.88 MB: AMD Zen 4 spec
            "l1i": 90 * 32 * 1024,  # 32 KB L1 instruction per core x 90 cores = 2.88 MB: AMD Zen 4 spec
            "l2": 90 * 1024 * 1024,  # 1 MB L2 per core x 90 cores = 90 MB: AMD Zen 4 spec
            "l3": 384 * 1024 * 1024,  # 384 MB L3 (12 CCDs x 32 MB): AMD Zen 4 spec
        },
    ),
    ram_bytes=720 * (1024**3),  # 720 GiB (773,094,113,280 bytes): GCP C3D spec sheet (c3d-standard-180)
    gpu=None,
    gpu_count=0,
    network=Network(
        bandwidth_bytes_per_sec=12.5e9,  # 100 Gbps (12.5 GB/s) default egress: GCP C3D networking documentation
    ),
    price=Price(
        on_demand_usd_per_hour=8.1718,  # $8.1718/hour on-demand: https://cloud.google.com/compute/all-pricing (us-central1, C3D)
        spot_usd_per_hour=1.9776,  # $1.9776/hour Spot: https://cloud.google.com/compute/all-pricing (us-central1, C3D)
    ),
)


C3D_HIGHMEM_180 = Machine(
    name="c3d-highmem-180",
    description="GCP C3D high-memory compute instance with AMD EPYC Genoa processors (180 vCPUs, 1.4 TB RAM)",
    cpu=CPU(
        model="AMD EPYC 9B14",  # GCP C3D machine family spec sheet (AMD 4th Gen Genoa custom CPU)
        cores=90,  # 90 physical cores: GCP C3D family spec (180 vCPUs / 2 threads per core)
        threads=180,  # 180 vCPUs: GCP C3D family spec sheet (c3d-highmem-180)
        simd="avx512",  # AMD Zen 4 (Genoa) architecture whitepaper: native AVX-512
        memory_bandwidth_bytes_per_sec=460.8e9,  # 460.8 GB/s (estimate: full single-socket AMD EPYC 9004 peak: 12 channels DDR5-4800, 12 * 8 * 4.8 GB/s)
        cache_sizes={
            "l1d": 90 * 32 * 1024,  # 32 KB L1 data per core x 90 cores = 2.88 MB: AMD Zen 4 spec
            "l1i": 90 * 32 * 1024,  # 32 KB L1 instruction per core x 90 cores = 2.88 MB: AMD Zen 4 spec
            "l2": 90 * 1024 * 1024,  # 1 MB L2 per core x 90 cores = 90 MB: AMD Zen 4 spec
            "l3": 384 * 1024 * 1024,  # 384 MB L3 (12 CCDs x 32 MB): AMD Zen 4 spec
        },
    ),
    ram_bytes=1440 * (1024**3),  # 1,440 GiB (1,546,188,226,560 bytes): GCP C3D spec sheet (c3d-highmem-180)
    gpu=None,
    gpu_count=0,
    network=Network(
        bandwidth_bytes_per_sec=12.5e9,  # 100 Gbps (12.5 GB/s) default egress: GCP C3D networking documentation
    ),
    price=Price(
        on_demand_usd_per_hour=11.0223,  # $11.0223/hour on-demand: https://cloud.google.com/compute/all-pricing (us-central1, C3D)
        spot_usd_per_hour=2.6674,  # $2.6674/hour Spot: https://cloud.google.com/compute/all-pricing (us-central1, C3D)
    ),
)


A3_HIGHGPU_8G = Machine(
    name="a3-highgpu-8g",
    description="GCP A3 accelerator instance with 8x NVIDIA H100 80GB SXM5 GPUs",
    cpu=CPU(
        model="Intel Xeon Platinum 8480C",  # GCP A3 machine family spec sheet (Intel 4th Gen Sapphire Rapids)
        cores=104,  # 104 physical cores: GCP A3 family spec (208 vCPUs / 2 threads per core)
        threads=208,  # 208 vCPUs: GCP A3 family spec sheet (a3-highgpu-8g)
        simd="avx512",  # Intel Sapphire Rapids architecture whitepaper: AVX-512 and AMX support
        memory_bandwidth_bytes_per_sec=614.4e9,  # 614.4 GB/s (estimate: dual-socket Intel Sapphire Rapids 16 channels DDR5-4800 peak: 16 * 8 * 4.8 GB/s)
        cache_sizes={
            "l1d": 104 * 48 * 1024,  # 48 KB L1 data per core x 104 cores = 4.99 MB: Intel Sapphire Rapids spec
            "l1i": 104 * 32 * 1024,  # 32 KB L1 instruction per core x 104 cores = 3.33 MB: Intel Sapphire Rapids spec
            "l2": 104 * 2 * 1024 * 1024,  # 2 MB L2 per core x 104 cores = 208 MB: Intel Sapphire Rapids spec
            "l3": 210 * 1024 * 1024,  # 210 MB shared L3 (2x 105 MB): Intel Sapphire Rapids spec
        },
    ),
    ram_bytes=1872 * (1024**3),  # 1,872 GiB (2,010,044,276,736 bytes): GCP A3 family spec sheet (a3-highgpu-8g)
    gpu=GPU(
        name="NVIDIA H100 80GB HBM3",  # NVIDIA Hopper H100 SXM5 80GB GPU
        memory_bytes=80 * (1024**3),  # 80 GiB (85,899,345,920 bytes) HBM3: NVIDIA H100 SXM5 datasheet
        memory_bandwidth_bytes_per_sec=3_350.0e9,  # 3,350 GB/s: NVIDIA H100 SXM5 datasheet (HBM3 memory bandwidth)
        peak_compute_flops=ComputeByDtype(
            bf16=989.0e12,  # 989 TFLOPS dense tensor: NVIDIA H100 SXM5 datasheet
            fp8=1_979.0e12,  # 1,979 TFLOPS dense tensor: NVIDIA H100 SXM5 datasheet
            fp32=67.0e12,  # 67 TFLOPS vector / standard: NVIDIA H100 SXM5 datasheet
        ),
        sm_count=132,  # 132 SMs: NVIDIA H100 SXM5 datasheet / GH100 whitepaper
        pcie_gen=5,  # PCIe Gen 5 host interface: NVIDIA H100 SXM5 datasheet
        pcie_lanes=16,  # 16 lanes: NVIDIA H100 SXM5 datasheet
        pcie_bandwidth_bytes_per_sec=63.0e9,  # ~63.0 GB/s each way: PCI-SIG PCIe 5.0 x16 theoretical bandwidth
    ),
    gpu_count=8,  # 8 GPUs: GCP A3 family spec sheet (a3-highgpu-8g)
    storage=LocalStorage(
        devices=16,  # 16 Local NVMe SSDs: GCP A3 family spec sheet (a3-highgpu-8g)
        capacity_bytes=6_000 * (10**9),  # 6,000 GB (16 x 375 GB): GCP Local SSD documentation
        read_bandwidth_bytes_per_sec=10.56e9,  # 10.56 GB/s aggregate (estimate: GCP Local SSD NVMe performance sheet, 16 x 660 MB/s)
        write_bandwidth_bytes_per_sec=5.60e9,  # 5.60 GB/s aggregate (estimate: GCP Local SSD NVMe performance sheet, 16 x 350 MB/s)
        read_iops=2_720_000.0,  # 2,720,000 IOPS aggregate (estimate: GCP Local SSD NVMe performance sheet, 16 x 170,000 IOPS)
        write_iops=1_440_000.0,  # 1,440,000 IOPS aggregate (estimate: GCP Local SSD NVMe performance sheet, 16 x 90,000 IOPS)
    ),
    network=Network(
        bandwidth_bytes_per_sec=125.0e9,  # 1,000 Gbps (125 GB/s): GCP A3 networking specification (GPUDirect RDMA + host net)
    ),
    price=Price(
        on_demand_usd_per_hour=88.49,  # $88.49/hour on-demand: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, A3 family)
        spot_usd_per_hour=50.32,  # $50.32/hour Spot: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, A3 family)
    ),
)


A2_HIGHGPU_1G = Machine(
    name="a2-highgpu-1g",
    description="GCP A2 accelerator instance with 1x NVIDIA A100 40GB SXM4 GPU",
    cpu=CPU(
        model="Intel Xeon Scalable (Cascade Lake)",  # GCP A2 machine family spec sheet (Intel 2nd Gen Cascade Lake)
        cores=6,  # 6 physical cores: GCP A2 family spec (12 vCPUs / 2 threads per core)
        threads=12,  # 12 vCPUs: GCP A2 family spec sheet (a2-highgpu-1g)
        simd="avx512",  # Intel Cascade Lake architecture whitepaper: AVX-512 support
        memory_bandwidth_bytes_per_sec=100.0e9,  # 100 GB/s (estimate: proportional share of 6-channel DDR4-2933)
        cache_sizes={
            "l1d": 6 * 32 * 1024,  # 32 KB L1 data per core x 6 cores = 0.19 MB: Intel Cascade Lake spec
            "l1i": 6 * 32 * 1024,  # 32 KB L1 instruction per core x 6 cores = 0.19 MB: Intel Cascade Lake spec
            "l2": 6 * 1024 * 1024,  # 1 MB L2 per core x 6 cores = 6 MB: Intel Cascade Lake spec
            "l3": 38 * 1024 * 1024,  # 38 MB shared L3: Intel Cascade Lake spec
        },
    ),
    ram_bytes=85 * (1024**3),  # 85 GiB (91,268,055,040 bytes): GCP A2 family spec sheet (a2-highgpu-1g)
    gpu=GPU(
        name="NVIDIA A100-SXM4-40GB",  # NVIDIA Ampere A100 SXM4 40GB GPU
        memory_bytes=40 * (1024**3),  # 40 GiB (42,949,672,960 bytes) HBM2: NVIDIA A100 SXM4 datasheet
        memory_bandwidth_bytes_per_sec=1_555.0e9,  # 1,555 GB/s: NVIDIA A100 SXM4 datasheet (HBM2 memory bandwidth)
        peak_compute_flops=ComputeByDtype(
            bf16=312.0e12,  # 312 TFLOPS dense tensor: NVIDIA A100 datasheet
            fp8=0.0,  # 0.0 FLOPS: NVIDIA Ampere architecture does not support FP8
            fp32=19.5e12,  # 19.5 TFLOPS vector / standard: NVIDIA A100 datasheet
        ),
        sm_count=108,  # 108 SMs: NVIDIA Ampere GA100 whitepaper / A100 datasheet
        pcie_gen=4,  # PCIe Gen 4 host interface: NVIDIA A100 SXM4 datasheet
        pcie_lanes=16,  # 16 lanes: NVIDIA A100 SXM4 datasheet
        pcie_bandwidth_bytes_per_sec=31.5e9,  # ~31.5 GB/s each way: PCI-SIG PCIe 4.0 x16 theoretical bandwidth
    ),
    gpu_count=1,  # 1 GPU: GCP A2 family spec sheet (a2-highgpu-1g)
    network=Network(
        bandwidth_bytes_per_sec=3.0e9,  # 24 Gbps (3.0 GB/s) maximum egress: GCP A2 networking documentation (a2-highgpu-1g)
    ),
    price=Price(
        on_demand_usd_per_hour=3.67,  # $3.67/hour on-demand: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, A2 family)
        spot_usd_per_hour=1.10,  # $1.10/hour Spot: https://cloud.google.com/compute/gpus-accelerators-pricing (us-central1, A2 family)
    ),
)


MAC_M4_PRO = Machine(
    name="mac-m4-pro",
    description="Apple Mac developer machine with M4 Pro chip (14 CPU cores, 20 GPU cores, unified memory)",
    cpu=CPU(
        model="Apple M4 Pro",  # Apple M4 Pro system-on-chip specification
        cores=14,  # 14 physical cores (10 performance + 4 efficiency): Apple M4 Pro specification
        threads=14,  # 14 threads (Apple Silicon does not use SMT): Apple M4 Pro specification
        simd="neon",  # ARMv9.2-A NEON & SME: Apple M4 Pro architecture documentation
        memory_bandwidth_bytes_per_sec=273.0e9,  # 273 GB/s unified memory: Apple M4 Pro specification (LPDDR5X-8533 bus)
        cache_sizes={
            "l1d": (10 * 128 + 4 * 64) * 1024,  # 128 KB P-core + 64 KB E-core L1 data = 1.54 MB: Apple M4 architecture analysis
            "l1i": (10 * 192 + 4 * 128) * 1024,  # 192 KB P-core + 128 KB E-core L1 inst = 2.43 MB: Apple M4 architecture analysis
            "l2": 20 * 1024 * 1024,  # 16 MB P-cluster L2 + 4 MB E-cluster L2 = 20 MB: Apple M4 Pro specification
            "l3": 24 * 1024 * 1024,  # 24 MB System Level Cache (SLC): Apple M4 Pro reverse engineering / analysis
        },
    ),
    ram_bytes=48 * (1024**3),  # 48 GiB (51,539,607,552 bytes) unified memory: Apple Mac developer workstation configuration
    gpu=GPU(
        name="Apple M4 Pro GPU",  # Apple M4 Pro integrated GPU (20 cores)
        memory_bytes=48 * (1024**3),  # 48 GiB unified memory shared with CPU: Apple M4 Pro specification
        memory_bandwidth_bytes_per_sec=273.0e9,  # 273 GB/s: Apple M4 Pro specification (shared unified memory bus)
        peak_compute_flops=ComputeByDtype(
            bf16=17.0e12,  # 17.0 TFLOPS (estimate: 2x FP32 throughput with 16-bit float)
            fp8=0.0,  # 0.0 FLOPS: Apple M4 Pro GPU does not feature native FP8 tensor cores
            fp32=8.5e12,  # 8.5 TFLOPS: Apple M4 Pro 20-core GPU (20 cores * 128 ALUs * 2 ops * ~1.9 GHz = ~9.7 TFLOPS theoretical peak, ~8.5 sustained)
        ),
        sm_count=0,  # 0 SMs: Apple Silicon GPU uses Apple core design (20 GPU cores, no NVIDIA SMs)
        pcie_gen=None,  # None: integrated on-die GPU (no PCIe bus)
        pcie_lanes=None,  # None: integrated on-die GPU
        pcie_bandwidth_bytes_per_sec=None,  # None: zero-copy unified memory fabric
    ),
    gpu_count=1,  # 1 integrated GPU: Apple M4 Pro specification
    storage=LocalStorage(
        devices=1,  # 1 internal Apple NVMe SSD
        capacity_bytes=1_000 * (10**9),  # 1,000 GB (1 TB internal SSD): Apple developer workstation configuration
        read_bandwidth_bytes_per_sec=5.5e9,  # 5.5 GB/s (estimate: Apple internal NVMe SSD benchmark read rate)
        write_bandwidth_bytes_per_sec=5.0e9,  # 5.0 GB/s (estimate: Apple internal NVMe SSD benchmark write rate)
        read_iops=200_000.0,  # 200,000 IOPS (estimate: modern PCIe 4.0 NVMe Apple SSD controller)
        write_iops=200_000.0,  # 200,000 IOPS (estimate: modern PCIe 4.0 NVMe Apple SSD controller)
    ),
    network=Network(
        bandwidth_bytes_per_sec=1.25e9,  # 10 Gbps (1.25 GB/s): 10GbE / Thunderbolt network
    ),
    price=Price(
        on_demand_usd_per_hour=0.0,  # $0.00/hour: local developer workstation hardware, no cloud rental fee
        spot_usd_per_hour=0.0,  # $0.00/hour: local developer workstation hardware, no cloud rental fee
    ),
    measured={
        "fp4_expert_decode_kernel_bytes_per_sec": 52.0e9,  # ~52 GB/s on laptop NEON with 10 threads (commit 5b99d74 and issue #269 microbenchmarks)
    },
)


CATALOG: dict[str, Machine] = {
    "g4-standard-48": G4_STANDARD_48,
    "g4-standard-96": G4_STANDARD_96,
    "c3d-standard-60": C3D_STANDARD_60,
    "c3d-standard-180": C3D_STANDARD_180,
    "c3d-highmem-180": C3D_HIGHMEM_180,
    "a3-highgpu-8g": A3_HIGHGPU_8G,
    "a2-highgpu-1g": A2_HIGHGPU_1G,
    "mac-m4-pro": MAC_M4_PRO,
}

ALIASES: dict[str, str] = {
    "a3-highgpu": "a3-highgpu-8g",
    "a2": "a2-highgpu-1g",
    "a2-highgpu": "a2-highgpu-1g",
    "m4-pro": "mac-m4-pro",
    "mac": "mac-m4-pro",
    "dev-mac": "mac-m4-pro",
}


def get_machine(name: str) -> Machine:
    """Retrieve a machine specification from the catalog by canonical name or alias."""
    key = name.strip().lower()
    if key in ("probe", "current"):
        return probe()
    resolved = ALIASES.get(key, key)
    if resolved in CATALOG:
        return CATALOG[resolved]
    available = sorted(set(list(CATALOG.keys()) + list(ALIASES.keys())))
    raise KeyError(f"unknown machine {name!r}; available: {', '.join(available)}")


# ---------------------------------------------------------------------------
# Hardware Probing
# ---------------------------------------------------------------------------

def _match_catalog(raw: dict[str, Any]) -> Optional[Machine]:
    """Match probed hardware to the closest reference machine in the catalog.

    Returns None if the GPU or CPU model does not match a known catalog entry.
    An unknown accelerator or host CPU must not borrow another machine's numbers.
    """
    if "gpu" in raw:
        gpu_name = str(raw["gpu"]).lower()
        gpu_count = int(raw.get("gpu_count") or 1)
        # Require "rtx pro 6000" explicitly: other GPUs (L4, T4, RTX A6000, etc.)
        # must not borrow G4's RTX PRO 6000 specs.
        if "rtx pro 6000" in gpu_name:
            if gpu_count == 1:
                return CATALOG["g4-standard-48"]
            return CATALOG["g4-standard-96"]
        if "h100" in gpu_name and gpu_count == 8:
            return CATALOG["a3-highgpu-8g"]
        if "a100" in gpu_name and gpu_count == 1:
            return CATALOG["a2-highgpu-1g"]
        if "m4 pro" in gpu_name or ("apple" in gpu_name and "m4" in gpu_name):
            return CATALOG["mac-m4-pro"]
        return None

    # CPU-only matching (match on CPU model string only; do not match by OS platform or vendor)
    cpu_model = str(raw.get("cpu", "")).lower()
    if "m4 pro" in cpu_model or ("apple" in cpu_model and "m4" in cpu_model):
        return CATALOG["mac-m4-pro"]
    if "9b14" in cpu_model or "genoa" in cpu_model:
        threads = int(raw.get("cpu_threads") or 0)
        ram = int(raw.get("ram_bytes") or 0)
        if threads >= 120 or ram >= 1000 * (1024**3):
            return CATALOG["c3d-highmem-180"] if ram >= 1000 * (1024**3) else CATALOG["c3d-standard-180"]
        return CATALOG["c3d-standard-60"]
    if "9b45" in cpu_model or "turin" in cpu_model or "9005" in cpu_model:
        threads = int(raw.get("cpu_threads") or 0)
        if threads >= 72:
            return CATALOG["g4-standard-96"]
        return CATALOG["g4-standard-48"]

    return None


def probe(device: Optional[Any] = None) -> Machine:
    """Build the current machine's entry from plan.probe_hardware,
    with the catalog filling in what can't be probed (bandwidths, prices),
    matched by GPU and CPU model.

    An unmatched GPU or CPU is given None for bandwidth, FLOPS and price.
    Where machine shape (thread count, GPU count) differs from catalog,
    shape-level attributes (memory bandwidth share, storage, network, price) are None.
    A probed machine does not inherit benchmark measurements from catalog entries.
    """
    if device is None:
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
        except (ImportError, Exception):
            pass

    raw = probe_hardware(device)
    matched = _match_catalog(raw)

    probed_threads = int(raw.get("cpu_threads") or (matched.cpu.threads if matched else 1))
    probed_gpu_count = int(raw.get("gpu_count") or 1) if "gpu" in raw else 0

    # Shape match requires matching thread count and GPU count.
    has_shape_match = (
        matched is not None
        and probed_threads == matched.cpu.threads
        and probed_gpu_count == matched.gpu_count
    )

    if matched and probed_threads == matched.cpu.threads:
        probed_cores = matched.cpu.cores
    else:
        # Do not assume SMT: non-matching thread counts report probed logical threads as cores
        probed_cores = probed_threads

    probed_simd = str(raw.get("simd") or (matched.cpu.simd if matched else "unknown"))
    probed_cpu_model = str(raw.get("cpu") or (matched.cpu.model if matched else "unknown"))
    probed_ram = int(raw.get("ram_bytes") or (matched.ram_bytes if matched else 0))

    # CPU memory bandwidth is a shape-level attribute (proportional share of socket)
    cpu_mem_bw = matched.cpu.memory_bandwidth_bytes_per_sec if has_shape_match else None
    cache_sizes = matched.cpu.cache_sizes if matched else None

    cpu = CPU(
        model=probed_cpu_model,
        cores=probed_cores,
        threads=probed_threads,
        simd=probed_simd,
        memory_bandwidth_bytes_per_sec=cpu_mem_bw,
        cache_sizes=cache_sizes,
    )

    if "gpu" in raw:
        probed_gpu_name = str(raw["gpu"])
        probed_gpu_mem = int(raw.get("gpu_memory_bytes") or (matched.gpu.memory_bytes if (matched and matched.gpu) else 0))

        if matched and matched.gpu:
            gpu = GPU(
                name=probed_gpu_name,
                memory_bytes=probed_gpu_mem,
                memory_bandwidth_bytes_per_sec=matched.gpu.memory_bandwidth_bytes_per_sec,
                peak_compute_flops=matched.gpu.peak_compute_flops,
                sm_count=matched.gpu.sm_count,
                pcie_gen=matched.gpu.pcie_gen,
                pcie_lanes=matched.gpu.pcie_lanes,
                pcie_bandwidth_bytes_per_sec=matched.gpu.pcie_bandwidth_bytes_per_sec,
            )
        else:
            # Unknown GPU: do not borrow another machine's bandwidth, FLOPS or SM count
            gpu = GPU(
                name=probed_gpu_name,
                memory_bytes=probed_gpu_mem,
                memory_bandwidth_bytes_per_sec=None,
                peak_compute_flops=None,
                sm_count=None,
                pcie_gen=None,
                pcie_lanes=None,
                pcie_bandwidth_bytes_per_sec=None,
            )
        gpu_count = probed_gpu_count
    else:
        gpu = None
        gpu_count = 0

    if has_shape_match:
        name = f"probed:{matched.name}"
        desc = f"Current machine probed on {device}, calibrated from {matched.name}"
        storage = matched.storage
        network = matched.network
        price = matched.price
    elif matched:
        name = f"probed:{matched.name}"
        desc = f"Current machine probed on {device}, calibrated from {matched.name} (shape differs)"
        storage = None
        network = None
        price = None
    else:
        name = f"probed:{probed_cpu_model.split()[0].lower() if probed_cpu_model else 'unknown'}"
        desc = f"Current machine probed on {device} (unmatched in catalog)"
        storage = None
        network = None
        price = None

    return Machine(
        name=name,
        description=desc,
        cpu=cpu,
        ram_bytes=probed_ram,
        gpu=gpu,
        gpu_count=gpu_count,
        storage=storage,
        network=network,
        price=price,
        measured=None,  # Do not inherit catalog benchmark measurements onto this host
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="vXPU hardware model schema and machine catalog")
    parser.add_argument("--machine", help="Machine name to inspect (or 'probe' / 'current')")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    if args.machine:
        try:
            machine = get_machine(args.machine)
        except KeyError as e:
            sys.exit(f"Error: {e}")
        if args.json:
            print(machine.to_json())
        else:
            print(machine.summary())
    else:
        if args.json:
            out = {k: v.to_dict() for k, v in CATALOG.items()}
            print(json.dumps(out, indent=2))
        else:
            print("=" * 72)
            print("vXPU Hardware Catalog")
            print("=" * 72)
            for k, m in CATALOG.items():
                print(m.summary())
                print("-" * 72)
            current = probe()
            print("Current Probed Hardware:")
            print(current.summary())
            print("=" * 72)


if __name__ == "__main__":
    main()
