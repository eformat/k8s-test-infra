// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
// SPDX-License-Identifier: Apache-2.0
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
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/stretchr/testify/require"

	"github.com/NVIDIA/k8s-test-infra/internal/fsutil"
	"github.com/NVIDIA/k8s-test-infra/pkg/gpu/mocknvml/engine"
)

// capsTreeRoot derives the tree root staging writes to from the overrides
// path a test pointed the engine at: the document lives at <root>/config/.
func capsTreeRoot(t *testing.T, overrides string) string {
	t.Helper()
	return filepath.Dir(filepath.Dir(overrides))
}

// capsMinors reads back the staged mig-minors table as the name→minor map a
// consumer resolves capabilities through.
func capsMinors(t *testing.T, root string) map[string]int {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, migCapsMinorsPath))
	require.NoError(t, err)
	minors := make(map[string]int)
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		fields := strings.Fields(line)
		require.Len(t, fields, 2, "malformed mig-minors line %q", line)
		minor, err := strconv.Atoi(fields[1])
		require.NoError(t, err)
		minors[fields[0]] = minor
	}
	return minors
}

// capsNodesSupported reports whether this environment may create character
// nodes at all — an unprivileged host cannot, and a test that cannot mknod
// still gets the table and the access files, which is what the NVML contract
// covers. The node assertions are what the DRA flow additionally needs, and
// they run wherever nodes may exist.
func capsNodesSupported(t *testing.T) bool {
	t.Helper()
	probe := filepath.Join(t.TempDir(), "nvidia-cap-probe")
	if err := syscall.Mknod(probe, syscall.S_IFCHR|0o600, int(migCapsMkdev(migCapsDefaultMajor, 99))); err != nil {
		return false
	}
	return os.Remove(probe) == nil
}

// The whole point: a partition a caller just created through NVML is fully
// usable the moment the call returns — its access file, its mig-minors line
// and its guarding node are all in place, and its teardown takes them back.
func TestMIGCaps_StageOnCreateUnstageOnDestroy(t *testing.T) {
	base := t.TempDir()
	overrides := filepath.Join(base, "config", "overrides.yaml")
	device := bootMIGEngine(t, migOffConfig, overrides)
	root := capsTreeRoot(t, overrides)

	minor, ret := engine.GetEngine().LookupConfigurableDevice(device).GetMinorNumber()
	require.Equal(t, nvml.SUCCESS, ret)

	ret, _ = migSetMode(device, nvml.DEVICE_MIG_ENABLE)
	require.Equal(t, nvml.SUCCESS, ret)

	gi, ret := migCreateGpuInstance(device, nvml.GPU_INSTANCE_PROFILE_1_SLICE, nil)
	require.Equal(t, nvml.SUCCESS, ret)
	giInfo, _, ret := engine.GetEngine().GpuInstanceGetInfo(gi)
	require.Equal(t, nvml.SUCCESS, ret)

	giName := migCapsGPUInstanceName(minor, giInfo.Id)
	minors := capsMinors(t, root)
	require.Contains(t, minors, giName, "the created GPU instance has no mig-minors entry")
	require.Equal(t, migCapsConfigMinor, minors["config"], "the node-wide entries seed the table")
	require.Equal(t, migCapsMonitorMinor, minors["monitor"])

	// The access file agrees with the table on the minor, since a consumer
	// reads the minor from either and they must answer identically.
	access := migCapsAccessPath(root, giName)
	data, err := os.ReadFile(access)
	require.NoError(t, err)
	require.Contains(t, string(data), fmt.Sprintf("DeviceFileMinor: %d", minors[giName]))
	require.Contains(t, string(data), fmt.Sprintf("DeviceFileMode: %d", migCapsFileMode))

	if capsNodesSupported(t) {
		var st syscall.Stat_t
		node := filepath.Join(root, migCapsDevDir, fmt.Sprintf("nvidia-cap%d", minors[giName]))
		require.NoError(t, syscall.Stat(node, &st), "no /dev node staged for the created GPU instance")
		require.Equal(t, syscall.S_IFCHR, st.Mode&syscall.S_IFMT, "the staged node is not a character device")
	}

	ci, ret := migCreateComputeInstance(gi, nvml.COMPUTE_INSTANCE_PROFILE_1_SLICE, nil)
	require.Equal(t, nvml.SUCCESS, ret)
	ciInfo, _, _, ret := engine.GetEngine().ComputeInstanceGetInfo(ci)
	require.Equal(t, nvml.SUCCESS, ret)

	ciName := migCapsComputeInstanceName(minor, giInfo.Id, ciInfo.Id)
	require.Contains(t, capsMinors(t, root), ciName)

	require.Equal(t, nvml.SUCCESS, migDestroyComputeInstance(ci))
	require.NotContains(t, capsMinors(t, root), ciName, "the destroyed compute instance is still staged")

	require.Equal(t, nvml.SUCCESS, migDestroyGpuInstance(gi))
	minors = capsMinors(t, root)
	require.NotContains(t, minors, giName, "the destroyed GPU instance is still staged")
	require.Contains(t, minors, "config", "the teardown keeps the node-wide entries")
}

// A disabled board has no partitions, so it has no capability surface either:
// the mode change takes every entry of the board's back, and only the
// node-wide entries survive.
func TestMIGCaps_ModeDisableWipesDeviceCaps(t *testing.T) {
	base := t.TempDir()
	overrides := filepath.Join(base, "config", "overrides.yaml")
	device := bootMIGEngine(t, migOffConfig, overrides)
	root := capsTreeRoot(t, overrides)

	ret, _ := migSetMode(device, nvml.DEVICE_MIG_ENABLE)
	require.Equal(t, nvml.SUCCESS, ret)
	gi, ret := migCreateGpuInstance(device, nvml.GPU_INSTANCE_PROFILE_1_SLICE, nil)
	require.Equal(t, nvml.SUCCESS, ret)
	_, ret = migCreateComputeInstance(gi, nvml.COMPUTE_INSTANCE_PROFILE_1_SLICE, nil)
	require.Equal(t, nvml.SUCCESS, ret)

	ret, _ = migSetMode(device, nvml.DEVICE_MIG_DISABLE)
	require.Equal(t, nvml.SUCCESS, ret)

	for name := range capsMinors(t, root) {
		require.NotRegexp(t, `^gpu0/`, name, "a disabled board still stages %s", name)
	}
}

// The agent stages the declared layout's surface at boot, with minors already
// handed out. Carving at runtime must leave those entries exactly as they
// are — a consumer may already hold the nodes they name — and allocate only
// past them.
func TestMIGCaps_PreservesBootStagedMinors(t *testing.T) {
	base := t.TempDir()
	overrides := filepath.Join(base, "config", "overrides.yaml")
	root := capsTreeRoot(t, overrides)

	// A boot-time table for the declared board: one instance, the minors the
	// agent would have handed it, and the files behind them.
	require.NoError(t, os.MkdirAll(filepath.Join(root, filepath.Dir(migCapsMinorsPath)), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, migCapsMinorsPath),
		[]byte("config 1\nmonitor 2\ngpu0/gi0/access 3\ngpu0/gi0/ci0/access 4\n"), 0o644))
	for name, minor := range map[string]int{
		"gpu0/gi0/access":     3,
		"gpu0/gi0/ci0/access": 4,
	} {
		require.NoError(t, fsutil.Write(migCapsAccessPath(root, name),
			[]byte(migCapsCapabilityFile(minor)), 0o644))
	}

	device := bootMIGEngine(t, migDeclaredConfig, overrides)
	minor, ret := engine.GetEngine().LookupConfigurableDevice(device).GetMinorNumber()
	require.Equal(t, nvml.SUCCESS, ret)
	require.Equal(t, 0, minor, "the boot table this test stages names gpu0")

	// The declared board already carries three instances; carving a fourth is
	// the runtime path this patch exists for.
	gi, ret := migCreateGpuInstance(device, nvml.GPU_INSTANCE_PROFILE_1_SLICE, nil)
	require.Equal(t, nvml.SUCCESS, ret)
	giInfo, _, ret := engine.GetEngine().GpuInstanceGetInfo(gi)
	require.Equal(t, nvml.SUCCESS, ret)

	minors := capsMinors(t, root)
	require.Equal(t, 3, minors["gpu0/gi0/access"], "the boot table's minor was reassigned")
	require.Equal(t, 4, minors["gpu0/gi0/ci0/access"], "the boot table's minor was reassigned")
	require.Equal(t, 5, minors[migCapsGPUInstanceName(minor, giInfo.Id)],
		"the carved instance did not allocate past the boot table")

	data, err := os.ReadFile(migCapsAccessPath(root, "gpu0/gi0/access"))
	require.NoError(t, err)
	require.Contains(t, string(data), "DeviceFileMinor: 3", "the boot entry's access file was rewritten")
}

// A consumer resolves the caps major the way the kernel would tell it — by
// reading proc/devices — and the DRA driver points that read at this tree.
// Staging a capability leaves the tree answering, in the exact shape the
// driver's resolver matches.
func TestMIGCaps_ProcDevicesNamesCapsMajor(t *testing.T) {
	base := t.TempDir()
	overrides := filepath.Join(base, "config", "overrides.yaml")
	device := bootMIGEngine(t, migOffConfig, overrides)
	root := capsTreeRoot(t, overrides)

	ret, _ := migSetMode(device, nvml.DEVICE_MIG_ENABLE)
	require.Equal(t, nvml.SUCCESS, ret)
	_, ret = migCreateGpuInstance(device, nvml.GPU_INSTANCE_PROFILE_1_SLICE, nil)
	require.Equal(t, nvml.SUCCESS, ret)

	data, err := os.ReadFile(filepath.Join(root, migCapsProcDevices))
	require.NoError(t, err)
	require.Regexp(t, `(?s)Character devices:.*?[0-9]+ nvidia-caps\n.*Block devices:`, string(data),
		"proc/devices does not resolve the nvidia-caps major the way the driver reads it")
}
