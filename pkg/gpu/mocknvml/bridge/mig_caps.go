// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// The MIG capability surface, staged as mutations happen.
//
// Finding a MIG device through NVML is not enough to use it: a consumer must
// also open the capability device guarding the partition, and it finds that
// device through a set of files no NVML call returns — the procfs access file
// per partition, the mig-minors table binding each capability to a
// /dev/nvidia-caps node, the nodes themselves, and the proc/devices entry
// naming the major. nvidia-container-toolkit reads all four while building a
// container's CDI spec, and a partition missing from any of them fails the
// pod that asked for it.
//
// The agent stages this surface once, at boot, for the layout the config
// declares. Partitions created later — the ones a DRA driver carves when a
// claim arrives, or nvidia-smi on a running node — would otherwise have no
// capability files at all. This file stages them, from the same process that
// performs the mutation, inside the same NVML call: by the time
// nvmlDeviceCreateGpuInstance returns, the surface for the instance it
// created is complete. That placement is what makes the staging race-free —
// an out-of-band watcher, however prompt, would still be racing the
// consumer's very next read, and losing that race is the exact failure this
// surface exists to prevent.
//
// Staging is best-effort and never fails the NVML call that triggered it: the
// engine's state and the override document remain the authority on what
// exists, and a consumer that cannot be handed a usable capability fails
// loudly at CDI construction, which is the honest place to fail it. A tree
// this process cannot write (a read-only mount in a tool container) is
// skipped with a debug line, matching how the override writers treat it.
package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/NVIDIA/go-nvml/pkg/nvml"

	"github.com/NVIDIA/k8s-test-infra/internal/fsutil"
	"github.com/NVIDIA/k8s-test-infra/pkg/gpu/mocknvml/engine"
)

// Paths under the driver tree root — the directory holding config/ — where
// the capability surface lives. These mirror what the agent stages at boot
// and what nvidia-container-toolkit reads back, so a partition staged here is
// indistinguishable from one staged at boot.
const (
	migCapsDevDir      = "dev/nvidia-caps"
	migCapsMinorsPath  = "proc/driver/nvidia-caps/mig-minors"
	migCapsCapsDir     = "proc/driver/nvidia/capabilities"
	migCapsProcDevices = "proc/devices"
)

// The node-wide capabilities are reserved minors the driver hands out before
// any GPU's, matching the agent's boot-time table and the toolkit's
// expectations for a node that has MIG capability files at all.
const (
	migCapsConfigMinor  = 1
	migCapsMonitorMinor = 2

	// migCapsDefaultMajor is the char-device major for /dev/nvidia-caps on
	// every node this simulator has run on. It is only a fallback: the major
	// is read back off an already-staged node when there is one, so a tree
	// staged with a different major stays consistent.
	migCapsDefaultMajor = 236

	// migCapsFileMode is the access-file mode copied from a real MIG-enabled
	// node — 292 is 0444, the same value the agent's boot-time staging writes.
	migCapsFileMode = 292
)

// migCapsMu serializes staging. NVML mutations can arrive on concurrent
// goroutines of one process, and staging reads, edits and rewrites files
// shared across every mutation; the engine's own locks do not cover them.
var migCapsMu sync.Mutex

// migCapsRoot derives the driver tree root from the overrides resolution, the
// same one the mutation itself was recorded through: the document lives at
// <root>/config/overrides.yaml, and the capability surface at the tree's
// proc and dev subtrees beside it. Deriving it here means the writer of a
// capability and every reader of one can never disagree about which tree is
// authoritative.
func migCapsRoot() string {
	override := engine.ConfigOverridePath()
	if override == "" {
		return ""
	}
	return filepath.Dir(filepath.Dir(override))
}

// migCapsFor resolves the two coordinates every staging act needs: the tree
// root and the parent GPU's device minor — the N in /dev/nvidiaN, which is
// how capability names identify a GPU.
func migCapsFor(dev *engine.ConfigurableDevice) (string, int, bool) {
	root := migCapsRoot()
	if root == "" {
		debugLog("[MIG] no overrides document resolved, capability staging skipped\n")
		return "", 0, false
	}
	minor, ret := dev.GetMinorNumber()
	if ret != nvml.SUCCESS {
		warnLog("[MIG] capability staging: reading the device minor: %v\n", ret)
		return "", 0, false
	}
	return root, minor, true
}

// migCapsStageGPUInstance stages the capability surface for a freshly created
// GPU instance. Called after the engine mutation and its record have
// committed, still inside the NVML call, so the surface is complete before
// the caller can ask for the instance's capability file.
func migCapsStageGPUInstance(dev *engine.ConfigurableDevice, gi uint32) {
	root, minor, ok := migCapsFor(dev)
	if !ok {
		return
	}
	migCapsAdd(root, migCapsGPUInstanceName(minor, gi))
}

// migCapsRemoveGPUInstance unstages the capability of a destroyed GPU
// instance, keeping the table and the tree in step with the engine.
func migCapsRemoveGPUInstance(dev *engine.ConfigurableDevice, gi uint32) {
	root, minor, ok := migCapsFor(dev)
	if !ok {
		return
	}
	migCapsRemove(root, migCapsGPUInstanceName(minor, gi))
}

// migCapsStageComputeInstance stages the capability of a freshly created
// compute instance, under its GPU instance's entry.
func migCapsStageComputeInstance(dev *engine.ConfigurableDevice, gi, ci uint32) {
	root, minor, ok := migCapsFor(dev)
	if !ok {
		return
	}
	migCapsAdd(root, migCapsComputeInstanceName(minor, gi, ci))
}

// migCapsRemoveComputeInstance unstages the capability of a destroyed compute
// instance.
func migCapsRemoveComputeInstance(dev *engine.ConfigurableDevice, gi, ci uint32) {
	root, minor, ok := migCapsFor(dev)
	if !ok {
		return
	}
	migCapsRemove(root, migCapsComputeInstanceName(minor, gi, ci))
}

// migCapsWipeDevice unstages every capability of one GPU. Disabling MIG
// destroyed the board's whole partitioning; its capability surface goes with
// it. The node-wide entries are not a partition's and stay.
func migCapsWipeDevice(dev *engine.ConfigurableDevice) {
	root, minor, ok := migCapsFor(dev)
	if !ok {
		return
	}
	migCapsMu.Lock()
	defer migCapsMu.Unlock()

	table, err := migCapsLoadTable(root)
	if err != nil {
		warnLog("[MIG] capability staging for device gpu%d: %v\n", minor, err)
		return
	}
	prefix := fmt.Sprintf("gpu%d/", minor)
	kept := make([]string, 0, len(table.names))
	for _, name := range table.names {
		if strings.HasPrefix(name, prefix) {
			migCapsUnstageFiles(root, name, table.minors[name])
			delete(table.minors, name)
			continue
		}
		kept = append(kept, name)
	}
	table.names = kept
	if err := table.save(root); err != nil {
		warnLog("[MIG] capability staging for device gpu%d: %v\n", minor, err)
	}
}

// migCapsAdd stages one capability: the procfs access file, the /dev node
// guarding it, and the mig-minors line binding the two. A capability already
// in the table is repaired rather than reallocated — the table is the
// authority on a capability's minor, and reassigning a minor a consumer may
// already hold would hand the node guarding one partition to another.
func migCapsAdd(root, name string) {
	migCapsMu.Lock()
	defer migCapsMu.Unlock()

	table, err := migCapsLoadTable(root)
	if err != nil {
		warnLog("[MIG] capability staging for %s: %v\n", name, err)
		return
	}
	minor, staged := table.minors[name]
	if !staged {
		minor = table.nextMinor(root)
		table.minors[name] = minor
		table.names = append(table.names, name)
	}
	// Every step warns rather than aborts, and the table lands last: the line
	// is the contract a consumer resolves the capability by, so it only
	// appears once the files naming its minor are in place. A step that
	// failed leaves a tree a later staging of the same capability repairs.
	if err := migCapsStageFiles(root, name, minor); err != nil {
		warnLog("[MIG] capability staging for %s: %v\n", name, err)
	}
	migCapsEnsureProcDevices(root)
	if err := table.save(root); err != nil {
		warnLog("[MIG] capability staging for %s: %v\n", name, err)
	}
}

// migCapsRemove unstages one capability: the table line, the access file and
// the /dev node. A capability absent from the table has nothing to remove —
// this runs on teardown paths that may repeat.
func migCapsRemove(root, name string) {
	migCapsMu.Lock()
	defer migCapsMu.Unlock()

	table, err := migCapsLoadTable(root)
	if err != nil {
		warnLog("[MIG] capability staging for %s: %v\n", name, err)
		return
	}
	minor, ok := table.minors[name]
	if !ok {
		debugLog("[MIG] capability %s: not staged, nothing to remove\n", name)
		return
	}
	migCapsUnstageFiles(root, name, minor)
	delete(table.minors, name)
	if idx := slices.Index(table.names, name); idx >= 0 {
		table.names = slices.Delete(table.names, idx, idx+1)
	}
	if err := table.save(root); err != nil {
		warnLog("[MIG] capability staging for %s: %v\n", name, err)
	}
}

// migCapsTable is the mig-minors table: one line per capability, in the order
// the file carries them. Consumers look capabilities up by name, so the order
// is presentational; the minor each name maps to is the contract.
type migCapsTable struct {
	names  []string
	minors map[string]int
}

// migCapsLoadTable reads the current table. A tree with no table yet gets the
// node-wide entries, the state a boot with MIG disabled leaves behind and the
// first runtime carve builds on.
func migCapsLoadTable(root string) (*migCapsTable, error) {
	path := filepath.Join(root, migCapsMinorsPath)
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return &migCapsTable{
			names: []string{"config", "monitor"},
			minors: map[string]int{
				"config":  migCapsConfigMinor,
				"monitor": migCapsMonitorMinor,
			},
		}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	table := &migCapsTable{minors: make(map[string]int)}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		name, minorText, found := strings.Cut(line, " ")
		if !found {
			return nil, fmt.Errorf("%s: malformed line %q", path, line)
		}
		minor, err := strconv.Atoi(strings.TrimSpace(minorText))
		if err != nil {
			return nil, fmt.Errorf("%s: malformed line %q: %w", path, line, err)
		}
		table.names = append(table.names, name)
		table.minors[name] = minor
	}
	return table, nil
}

// save rewrites the table file, atomically, with every entry's current minor.
func (t *migCapsTable) save(root string) error {
	var b strings.Builder
	for _, name := range t.names {
		fmt.Fprintf(&b, "%s %d\n", name, t.minors[name])
	}
	path := filepath.Join(root, migCapsMinorsPath)
	if err := fsutil.Write(path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}

// nextMinor allocates the minor for a new capability: one past everything the
// table and the device directory already carry. Minors of live capabilities
// are never reassigned, so a node a consumer holds keeps guarding the
// partition it was handed for; scanning the device directory as well as the
// table keeps the allocation coherent even if a previous write left a node
// whose line never landed.
func (t *migCapsTable) nextMinor(root string) int {
	next := migCapsMonitorMinor + 1
	for _, minor := range t.minors {
		if minor >= next {
			next = minor + 1
		}
	}
	entries, err := os.ReadDir(filepath.Join(root, migCapsDevDir))
	if err != nil {
		return next
	}
	for _, entry := range entries {
		text := strings.TrimPrefix(entry.Name(), "nvidia-cap")
		if text == entry.Name() {
			continue
		}
		if minor, err := strconv.Atoi(text); err == nil && minor >= next {
			next = minor + 1
		}
	}
	return next
}

// migCapsStageFiles writes a capability's access file and its /dev node.
// Idempotent: files are rewritten in place, an existing node is kept.
func migCapsStageFiles(root, name string, minor int) error {
	access := migCapsAccessPath(root, name)
	if err := fsutil.Write(access, []byte(migCapsCapabilityFile(minor)), 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", access, err)
	}
	return migCapsStageNode(root, minor)
}

// migCapsStageNode creates the /dev/nvidia-caps node guarding a capability.
func migCapsStageNode(root string, minor int) error {
	dir := filepath.Join(root, migCapsDevDir)
	node := filepath.Join(dir, fmt.Sprintf("nvidia-cap%d", minor))
	if _, err := os.Stat(node); err == nil {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", dir, err)
	}
	major := migCapsMajor(dir)
	if err := syscall.Mknod(node, syscall.S_IFCHR|0o666, int(migCapsMkdev(uint64(major), uint64(minor)))); err != nil {
		return fmt.Errorf("mknod %s (major %d): %w", node, major, err)
	}
	return nil
}

// migCapsMajor reads the char-device major for nvidia-caps off an
// already-staged node, falling back to the major every node this simulator
// has run on uses. Reading it back keeps a tree staged with a different
// major self-consistent.
func migCapsMajor(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return migCapsDefaultMajor
	}
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "nvidia-cap") {
			continue
		}
		var st syscall.Stat_t
		if err := syscall.Stat(filepath.Join(dir, entry.Name()), &st); err != nil {
			continue
		}
		return int((uint64(st.Rdev) >> 8) & 0xfff)
	}
	return migCapsDefaultMajor
}

// migCapsUnstageFiles removes a capability's access file and /dev node.
// Anything already gone is fine — teardown paths may repeat.
func migCapsUnstageFiles(root, name string, minor int) {
	access := migCapsAccessPath(root, name)
	if err := os.Remove(access); err != nil && !errors.Is(err, os.ErrNotExist) {
		warnLog("[MIG] capability %s: removing %s: %v\n", name, access, err)
	}
	node := filepath.Join(root, migCapsDevDir, fmt.Sprintf("nvidia-cap%d", minor))
	if err := os.Remove(node); err != nil && !errors.Is(err, os.ErrNotExist) {
		warnLog("[MIG] capability %s: removing %s: %v\n", name, node, err)
	}
}

// migCapsAccessPath maps a capability name to its access file under
// capabilities/, mirroring nvidia-container-toolkit's MigCap.ProcPath: the
// node-wide capabilities live under mig/, a GPU's under gpu<N>/mig/.
func migCapsAccessPath(root, name string) string {
	gpu, rest, found := strings.Cut(name, "/")
	if !found {
		return filepath.Join(root, migCapsCapsDir, "mig", name)
	}
	return filepath.Join(root, migCapsCapsDir, gpu, "mig", rest)
}

// migCapsCapabilityFile renders an access file's body. Mode 292 is 0444 and
// Modify 1 is what the driver reports for a capability the caller may use;
// both are copied from a real MIG-enabled node, as in the agent's boot-time
// staging, so the two writers produce identical files.
func migCapsCapabilityFile(minor int) string {
	return fmt.Sprintf("DeviceFileMinor: %d\nDeviceFileMode: %d\nDeviceFileModify: 1\n", minor, migCapsFileMode)
}

// migCapsGPUInstanceName and migCapsComputeInstanceName spell capability
// names the way nvidia-container-toolkit's NewGPUInstanceCap and
// NewComputeInstanceCap do, so the table this staging writes is the table the
// toolkit reads.
func migCapsGPUInstanceName(gpuMinor int, gi uint32) string {
	return fmt.Sprintf("gpu%d/gi%d/access", gpuMinor, gi)
}

func migCapsComputeInstanceName(gpuMinor int, gi, ci uint32) string {
	return fmt.Sprintf("gpu%d/gi%d/ci%d/access", gpuMinor, gi, ci)
}

// migCapsEnsureProcDevices makes the tree's proc/devices name the nvidia-caps
// major. A consumer resolves the major the way the kernel would tell it — by
// reading proc/devices — and the DRA driver points that read at this tree, so
// a tree hosting capability nodes has to answer the same question.
// Callers hold migCapsMu, as every act of staging does.
func migCapsEnsureProcDevices(root string) {
	major := migCapsMajor(filepath.Join(root, migCapsDevDir))
	path := filepath.Join(root, migCapsProcDevices)
	data, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		body := fmt.Sprintf("Character devices:\n  %d nvidia-caps\n\nBlock devices:\n", major)
		if err := fsutil.Write(path, []byte(body), 0o644); err != nil {
			warnLog("[MIG] staging %s: %v\n", path, err)
		}
	case err != nil:
		warnLog("[MIG] staging %s: %v\n", path, err)
	default:
		if strings.Contains(string(data), " nvidia-caps\n") {
			return
		}
		lines := strings.SplitAfter(string(data), "\n")
		for i, line := range lines {
			if strings.TrimSpace(line) != "Character devices:" {
				continue
			}
			lines = slices.Insert(lines, i+1, fmt.Sprintf("  %d nvidia-caps\n", major))
			if err := fsutil.Write(path, []byte(strings.Join(lines, "")), 0o644); err != nil {
				warnLog("[MIG] staging %s: %v\n", path, err)
			}
			return
		}
		body := string(data) + fmt.Sprintf("\nCharacter devices:\n  %d nvidia-caps\n\nBlock devices:\n", major)
		if err := fsutil.Write(path, []byte(body), 0o644); err != nil {
			warnLog("[MIG] staging %s: %v\n", path, err)
		}
	}
}

// migCapsMkdev packs a char device's major and minor the way the kernel's
// makedev does, for nodes the old encoding cannot express either.
func migCapsMkdev(major, minor uint64) uint64 {
	return (minor & 0xff) | (major&0xfff)<<8 | (minor&^0xff)<<12 | (major&^0xfff)<<32
}
