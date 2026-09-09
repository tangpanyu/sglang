import tempfile
import sys
import unittest
from pathlib import Path

# This directory is intentionally not a Python package; make direct path-based
# test invocation work both from this directory and from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_trace_probe import StateTraceProbe, load_records, validate_records


class _FakeStorage:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class FakeTensor:
    """Tiny torch-like object so the probe can be tested without torch/GPU."""

    def __init__(self, values, *, ptr=4096, storage_ptr=None, offset=0):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = "torch.float32"
        self.device = "cuda:0"
        self._ptr = ptr
        self._storage_ptr = ptr if storage_ptr is None else storage_ptr
        self._offset = offset

    def stride(self):
        return (1,)

    def data_ptr(self):
        return self._ptr

    def storage_offset(self):
        return self._offset

    def untyped_storage(self):
        return _FakeStorage(self._storage_ptr)

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def reshape(self, value):
        assert value == -1
        return self

    def float(self):
        return FakeTensor(
            [float(value) for value in self.values],
            ptr=self._ptr,
            storage_ptr=self._storage_ptr,
            offset=self._offset,
        )

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            return self.values[item]
        start, stop, step = item.indices(len(self.values))
        values = self.values[item]
        return FakeTensor(
            values,
            ptr=self._ptr + start * 4,
            storage_ptr=self._storage_ptr,
            offset=self._offset + start,
        )


class StateTraceProbeTest(unittest.TestCase):
    def test_valid_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with StateTraceProbe(path, include_values=True, max_values=4) as probe:
                self.assertEqual(probe.small_list(FakeTensor([21, 22, 23])), [21, 22, 23])
                probe.owner(
                    space="recurrent",
                    resource_id=5,
                    state="FREE",
                    owner=None,
                    reason="initial inventory",
                )
                probe.owner(
                    space="recurrent",
                    resource_id=5,
                    state="ACTIVE",
                    owner="rid:A",
                    reason="alloc",
                )
                probe.capture(
                    "ALLOC",
                    rid="A",
                    forward_id="prefill-0",
                    req_row=7,
                    seq_len=4,
                    full_locs=[21, 22, 23, 24],
                    write_locs=[21, 22, 23, 24],
                    state_slot_virtual=5,
                )
                probe.capture(
                    "FORWARD_META",
                    forward_id="prefill-0",
                    rids=["A"],
                    req_rows=[7],
                    state_slots_virtual=[5],
                    state_slots_physical=[5],
                    out_cache_loc=[21, 22, 23, 24],
                    query_start_loc=[0, 4],
                    require_unique_writers=True,
                )
                before = FakeTensor([0.0, 0.0, 0.0], ptr=8192)
                after = FakeTensor([1.0, 2.0, 3.0], ptr=8192)
                probe.capture(
                    "STATE_BEFORE",
                    rid="A",
                    forward_id="prefill-0",
                    layer_id=1,
                    req_row=7,
                    state_slot_physical=5,
                    depth_before=0,
                    transition_id="A/prefill-0/layer-1",
                    input_ids=[101, 42, 17, 9],
                    tensor_values={"temporal_slot": before},
                )
                probe.capture(
                    "STATE_AFTER",
                    rid="A",
                    forward_id="prefill-0",
                    layer_id=1,
                    req_row=7,
                    state_slot_physical=5,
                    depth_after=4,
                    transition_id="A/prefill-0/layer-1",
                    expect_changed=["temporal_slot"],
                    tensor_values={"temporal_slot": after},
                )

                probe.owner(
                    space="recurrent",
                    resource_id=12,
                    state="CACHED",
                    owner="node:P256",
                    reason="prefix checkpoint",
                )
                src = FakeTensor([4.0, 5.0], ptr=12288)
                dst = FakeTensor([4.0, 5.0], ptr=16384)
                probe.capture(
                    "COW_PLAN",
                    rid="B",
                    transition_id="B/cow/12-8",
                    src_slot_physical=12,
                    dst_slot_physical=8,
                    src_owner_state="CACHED",
                    src_owner="node:P256",
                    tensor_values={"src_before": src},
                )
                probe.capture(
                    "COW_DONE",
                    rid="B",
                    transition_id="B/cow/12-8",
                    src_slot_physical=12,
                    dst_slot_physical=8,
                    src_owner_state="CACHED",
                    src_owner="node:P256",
                    tensor_values={"src_after": src, "dst_after": dst},
                )
                probe.owner(
                    space="recurrent",
                    resource_id=8,
                    state="ACTIVE",
                    owner="rid:B",
                    reason="cow destination",
                )

            result = validate_records(load_records(path))
            self.assertTrue(result.ok, result.errors)

    def test_rejects_unproven_state_transition(self):
        records = [
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 0,
                "event": "STATE_BEFORE",
                "rid": "A",
                "forward_id": "f0",
                "layer_id": 1,
                "req_row": 7,
                "state_slot_physical": 5,
                "depth_before": 0,
                "transition_id": "A/f0/1",
                "input_ids": [1],
            },
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 1,
                "event": "STATE_AFTER",
                "rid": "A",
                "forward_id": "f0",
                "layer_id": 1,
                "req_row": 7,
                "state_slot_physical": 5,
                "depth_after": 1,
                "transition_id": "A/f0/1",
            },
        ]
        result = validate_records(records)
        self.assertFalse(result.ok)
        self.assertTrue(any("persistent tensor" in e for e in result.errors))

    def test_cow_signatures_must_match_snapshots(self):
        records = [
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 0,
                "event": "OWNER",
                "space": "recurrent",
                "resource_id": 12,
                "state": "CACHED",
                "owner": "node:P256",
                "reason": "prefix checkpoint",
            },
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 1,
                "event": "COW_PLAN",
                "rid": "B",
                "transition_id": "B/cow",
                "src_slot_physical": 12,
                "dst_slot_physical": 8,
                "src_owner_state": "CACHED",
                "src_owner": "node:P256",
                "tensors": {
                    "src_before": {
                        "shape": [2],
                        "stride": [1],
                        "data_ptr": 100,
                        "sample_sha256": "actual-before",
                    }
                },
            },
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 2,
                "event": "COW_DONE",
                "rid": "B",
                "transition_id": "B/cow",
                "src_slot_physical": 12,
                "dst_slot_physical": 8,
                "src_owner_state": "CACHED",
                "src_owner": "node:P256",
                "src_before_sig": "declared-but-false",
                "tensors": {
                    "src_after": {
                        "shape": [2],
                        "stride": [1],
                        "data_ptr": 100,
                        "sample_sha256": "actual-source",
                    },
                    "dst_after": {
                        "shape": [2],
                        "stride": [1],
                        "data_ptr": 200,
                        "sample_sha256": "actual-destination",
                    },
                },
            },
        ]
        result = validate_records(records)
        self.assertFalse(result.ok)
        self.assertTrue(any("declared src_before_sig" in e for e in result.errors))

    def test_rejects_cached_to_active_alias(self):
        records = [
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 0,
                "event": "OWNER",
                "space": "recurrent",
                "resource_id": 12,
                "state": "CACHED",
                "owner": "node:P256",
                "reason": "prefix checkpoint",
            },
            {
                "schema": "sglang-state-trace/v1",
                "event_seq": 1,
                "event": "OWNER",
                "space": "recurrent",
                "resource_id": 12,
                "state": "ACTIVE",
                "owner": "rid:B",
                "reason": "illegal alias",
            },
        ]
        result = validate_records(records)
        self.assertFalse(result.ok)
        self.assertTrue(any("illegal owner transition" in e for e in result.errors))

    def test_append_continues_event_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with StateTraceProbe(path) as probe:
                probe.capture("NOTE", value="first")
            with StateTraceProbe(path, append=True) as probe:
                probe.capture("NOTE", value="second")
            records = load_records(path)
            self.assertEqual([record["event_seq"] for record in records], [0, 1])
            self.assertTrue(validate_records(records).ok)

    def test_append_separates_missing_final_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                '{"schema":"sglang-state-trace/v1","event_seq":0,"event":"NOTE"}',
                encoding="utf-8",
            )
            with StateTraceProbe(path, append=True) as probe:
                probe.capture("NOTE", value="second")
            records = load_records(path)
            self.assertEqual([record["event_seq"] for record in records], [0, 1])

    def test_empty_trace_is_rejected(self):
        result = validate_records([])
        self.assertFalse(result.ok)
        self.assertIn("trace is empty", result.errors)

    def test_missing_structural_field_is_rejected(self):
        result = validate_records(
            [
                {
                    "schema": "sglang-state-trace/v1",
                    "event_seq": 0,
                    "event": "FORWARD_META",
                    "rids": ["A"],
                    "req_rows": [7],
                }
            ]
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("state_slots_physical" in e for e in result.errors))

    def test_forward_meta_rejects_duplicate_rid_and_bad_endpoint(self):
        result = validate_records(
            [
                {
                    "schema": "sglang-state-trace/v1",
                    "event_seq": 0,
                    "event": "FORWARD_META",
                    "forward_id": "decode-1",
                    "rids": ["A", "A"],
                    "req_rows": [7, 8],
                    "state_slots_virtual": [5, 6],
                    "state_slots_physical": [5, 6],
                    "out_cache_loc": [25, 26],
                    "query_start_loc": [0, 1, 99],
                }
            ]
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("rids must be unique" in e for e in result.errors))
        self.assertTrue(any("len(out_cache_loc)" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()
