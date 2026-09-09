#!/usr/bin/env python3
"""Small, dependency-free JSONL probe for SGLang serving-state debugging.

The probe deliberately does not import torch.  Pass real torch tensors from a
debug console; tensor metadata stays on the host, while sampled values are
copied to the host only when ``include_values=True``.

This is a debugger-side teaching tool, not a production profiler.  Value
sampling synchronizes the device and therefore perturbs latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SCHEMA = "sglang-state-trace/v1"
OWNER_STATES = {"FREE", "ACTIVE", "CACHED"}
OWNER_TRANSITIONS = {
    ("FREE", "ACTIVE"),
    ("ACTIVE", "FREE"),
    ("ACTIVE", "CACHED"),
    ("CACHED", "FREE"),
}
RESERVED_RECORD_FIELDS = {"schema", "event_seq", "time_ns", "pid", "event", "tensors"}


def _call_or_value(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, float) and not value.is_integer():
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Enum-like values are common in ForwardMode.  This conversion is host-only.
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return value.name
    raise TypeError(
        f"{type(value).__name__} is not JSON-safe. "
        "For a small tensor call probe.small_list(tensor) explicitly; "
        "automatic tensor-to-CPU conversion is intentionally disabled."
    )


def _flat_tensor(tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    x = detach() if callable(detach) else tensor
    reshape = getattr(x, "reshape", None)
    if not callable(reshape):
        raise TypeError("tensor-like object must provide reshape(-1)")
    return reshape(-1)


def _to_host_list(tensor: Any, *, cast_float: bool = False) -> list[Any]:
    x = tensor
    if cast_float:
        float_fn = getattr(x, "float", None)
        if callable(float_fn):
            x = float_fn()
    cpu_fn = getattr(x, "cpu", None)
    if callable(cpu_fn):
        x = cpu_fn()
    tolist = getattr(x, "tolist", None)
    if not callable(tolist):
        raise TypeError("tensor-like object must provide cpu().tolist()")
    value = tolist()
    return value if isinstance(value, list) else [value]


def _sample_values(tensor: Any, max_values: int) -> list[Any]:
    if max_values <= 0:
        return []
    flat = _flat_tensor(tensor)
    numel = _as_int(_call_or_value(flat, "numel"), 0) or 0
    if numel == 0:
        return []
    step = max(1, (numel + max_values - 1) // max_values)
    sampled = flat[::step][:max_values]
    return _to_host_list(sampled, cast_float=True)


def _hash_sample(values: list[Any]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _next_sequence(path: Path) -> int:
    """Return the next event sequence for an append-mode trace.

    Append mode must not restart at zero: duplicate sequence numbers make a
    merged trace ambiguous and are rejected by ``validate_records``.  Parse the
    existing JSONL here so a malformed/partial file fails before a new event is
    added to it.
    """

    if not path.exists() or path.stat().st_size == 0:
        return 0
    last = -1
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"cannot append to invalid JSON on line {line_no}: {exc}"
                ) from exc
            seq = _as_int(record.get("event_seq")) if isinstance(record, Mapping) else None
            if seq is None or seq < 0:
                raise ValueError(
                    f"cannot append: line {line_no} has invalid event_seq"
                )
            last = max(last, seq)
    return last + 1


def tensor_snapshot(
    tensor: Any,
    *,
    include_values: bool = False,
    max_values: int = 16,
) -> dict[str, Any]:
    """Capture pointer/layout metadata and optionally a deterministic sample.

    ``data_ptr`` is the first element of this view; ``storage_ptr`` is the base
    allocation when the tensor API exposes it.  For a slot-specific address,
    pass a basic-index view such as ``temporal[slot]``.  Do not pass an
    advanced-index gather such as ``temporal[indices]`` and interpret its
    pointer as the persistent slot address.
    """

    shape = _call_or_value(tensor, "shape", ())
    stride = _call_or_value(tensor, "stride", ())
    data_ptr = _as_int(_call_or_value(tensor, "data_ptr"))
    storage_offset = _as_int(_call_or_value(tensor, "storage_offset"), 0)
    storage_ptr = None
    try:
        storage = tensor.untyped_storage()
        storage_ptr = _as_int(storage.data_ptr())
    except (AttributeError, RuntimeError, TypeError):
        pass

    result: dict[str, Any] = {
        "shape": [int(x) for x in shape],
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "device": str(getattr(tensor, "device", "unknown")),
        "stride": [int(x) for x in stride],
        "numel": _as_int(_call_or_value(tensor, "numel"), 0),
        "data_ptr": data_ptr,
        "storage_ptr": storage_ptr,
        "storage_offset": storage_offset,
    }
    if include_values:
        values = _sample_values(tensor, max_values)
        result["sample"] = values
        result["sample_count"] = len(values)
        result["sample_sha256"] = _hash_sample(values)
    return result


class StateTraceProbe:
    """Append structured state observations to one JSONL file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        include_values: bool = False,
        max_values: int = 16,
        append: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_values = bool(include_values)
        parsed_max_values = _as_int(max_values)
        if parsed_max_values is None or parsed_max_values <= 0:
            raise ValueError("max_values must be a positive integer")
        self.max_values = parsed_max_values
        self._seq = _next_sequence(self.path) if append else 0
        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")
        if append and self.path.stat().st_size > 0:
            # A hand-written JSONL file may omit its final newline.  Separate
            # the next record before appending, otherwise two JSON objects
            # become one invalid line.
            with self.path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) not in (b"\n", b"\r"):
                    self._fh.write("\n")
                    self._fh.flush()

    def __enter__(self) -> "StateTraceProbe":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def small_list(self, tensor: Any, *, limit: int = 64) -> list[Any]:
        """Explicitly copy a small index/length tensor to the host.

        The size guard prevents an accidental full cache dump.  This operation
        synchronizes a CUDA tensor, so call it only at a deliberate breakpoint.
        """

        flat = _flat_tensor(tensor)
        numel = _as_int(_call_or_value(flat, "numel"), 0) or 0
        if numel > limit:
            raise ValueError(f"refusing to copy {numel} values; limit={limit}")
        return _to_host_list(flat)

    def signature(self, tensor: Any) -> Optional[str]:
        """Return the sampled-value signature used by COW checks."""

        if not self.include_values:
            return None
        return tensor_snapshot(
            tensor, include_values=True, max_values=self.max_values
        )["sample_sha256"]

    def capture(
        self,
        event: str,
        *,
        tensor_meta: Optional[Mapping[str, Any]] = None,
        tensor_values: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Record one observation.

        ``tensor_meta`` never copies values.  ``tensor_values`` records the same
        metadata and includes a small sample only when the probe was created
        with ``include_values=True``.
        """

        overlap = RESERVED_RECORD_FIELDS.intersection(fields)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"reserved record field(s) cannot be overridden: {names}")

        record: dict[str, Any] = {
            "schema": SCHEMA,
            "event_seq": self._seq,
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "event": str(event),
        }
        record.update({str(k): _json_safe(v) for k, v in fields.items()})

        tensors: dict[str, Any] = {}
        for name, tensor in (tensor_meta or {}).items():
            tensors[str(name)] = tensor_snapshot(tensor, include_values=False)
        for name, tensor in (tensor_values or {}).items():
            tensors[str(name)] = tensor_snapshot(
                tensor,
                include_values=self.include_values,
                max_values=self.max_values,
            )
        if tensors:
            record["tensors"] = tensors

        payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self._fh.write(payload)
        self._fh.flush()
        self._seq += 1
        return record

    def owner(
        self,
        *,
        space: str,
        resource_id: int,
        state: str,
        owner: Optional[str],
        reason: str,
        **fields: Any,
    ) -> dict[str, Any]:
        state = state.upper()
        if state not in OWNER_STATES:
            raise ValueError(f"invalid owner state: {state}")
        return self.capture(
            "OWNER",
            space=space,
            resource_id=int(resource_id),
            state=state,
            owner=owner,
            reason=reason,
            **fields,
        )


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
    return records


def _signature(record: Mapping[str, Any], tensor_name: str) -> Optional[str]:
    if not isinstance(tensor_name, str):
        return None
    tensors = record.get("tensors", {})
    if not isinstance(tensors, Mapping):
        return None
    try:
        snapshot = tensors.get(tensor_name, {})
    except (TypeError, AttributeError):
        return None
    return snapshot.get("sample_sha256") if isinstance(snapshot, Mapping) else None


def validate_records(records: Iterable[Mapping[str, Any]]) -> ValidationResult:
    records = list(records)
    errors: list[str] = []
    warnings: list[str] = []
    owner_by_resource: dict[tuple[str, int], tuple[str, Optional[str]]] = {}
    state_before: dict[str, Mapping[str, Any]] = {}
    cow_plans: dict[str, Mapping[str, Any]] = {}
    seen_seq: list[int] = []
    known_events = {
        "ALLOC",
        "FORWARD_META",
        "STATE_BEFORE",
        "STATE_AFTER",
        "CLEAR_BEFORE",
        "CLEAR_AFTER",
        "COW_PLAN",
        "COW_DONE",
        "OWNER",
    }

    if not records:
        return ValidationResult(errors=["trace is empty"], warnings=[])

    def require(rec: Mapping[str, Any], seq: int, *names: str) -> bool:
        missing = [name for name in names if name not in rec]
        if missing:
            errors.append(f"event {seq}: missing required field(s): {', '.join(missing)}")
            return False
        return True

    def integer_field(rec: Mapping[str, Any], seq: int, name: str, *, minimum: int | None = None) -> Optional[int]:
        value = _as_int(rec.get(name))
        if value is None or (minimum is not None and value < minimum):
            suffix = f" >= {minimum}" if minimum is not None else ""
            errors.append(f"event {seq}: {name} must be an integer{suffix}")
            return None
        return value

    def list_of_ints(
        value: Any,
        seq: int,
        name: str,
        *,
        minimum: int | None = None,
    ) -> Optional[list[int]]:
        if not isinstance(value, list):
            errors.append(f"event {seq}: {name} must be a list")
            return None
        result: list[int] = []
        for item in value:
            parsed = _as_int(item)
            if parsed is None or (minimum is not None and parsed < minimum):
                suffix = f" >= {minimum}" if minimum is not None else ""
                errors.append(f"event {seq}: {name} contains a non-integer{suffix}: {item!r}")
            else:
                result.append(parsed)
        return result

    def snapshot_map(rec: Mapping[str, Any], seq: int, name: str) -> Optional[Mapping[str, Any]]:
        tensors = rec.get("tensors")
        if not isinstance(tensors, Mapping) or not tensors:
            errors.append(f"event {seq}: {name} must contain a non-empty tensors mapping")
            return None
        return tensors

    def check_snapshot(snapshot: Any, seq: int, name: str) -> bool:
        if not isinstance(snapshot, Mapping):
            errors.append(f"event {seq}: {name} tensor snapshot must be an object")
            return False
        missing = [field for field in ("shape", "stride", "data_ptr") if field not in snapshot]
        if missing:
            errors.append(
                f"event {seq}: {name} tensor snapshot missing {', '.join(missing)}"
            )
            return False
        valid = True
        for field in ("shape", "stride"):
            value = snapshot.get(field)
            if not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in value
            ):
                errors.append(f"event {seq}: {name}.{field} must be a list of non-negative integers")
                valid = False
        pointer = snapshot.get("data_ptr")
        if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer < 0:
            errors.append(f"event {seq}: {name}.data_ptr must be a non-negative integer")
            valid = False
        return valid

    def check_identity_fields(rec: Mapping[str, Any], seq: int, *, include_forward: bool = True) -> None:
        if not isinstance(rec.get("rid"), str) or not rec.get("rid"):
            errors.append(f"event {seq}: rid must be a non-empty string")
        if include_forward and (not isinstance(rec.get("forward_id"), str) or not rec.get("forward_id")):
            errors.append(f"event {seq}: forward_id must be a non-empty string")

    for index, rec in enumerate(records, 1):
        if not isinstance(rec, Mapping):
            errors.append(f"record {index}: expected an object")
            seen_seq.append(-1)
            continue

        raw_seq = rec.get("event_seq")
        seq = _as_int(raw_seq, -1)
        seen_seq.append(seq if seq is not None else -1)
        event = rec.get("event")

        if rec.get("schema") != SCHEMA:
            errors.append(f"event {seq}: schema is not {SCHEMA}")
        if not isinstance(raw_seq, (int, float, str)) or _as_int(raw_seq) is None:
            errors.append(f"event {seq}: event_seq must be an integer")
        elif seq < 0:
            errors.append(f"event {seq}: event_seq must be non-negative")
        if not isinstance(event, str) or not event:
            errors.append(f"event {seq}: missing event name")
            continue
        if event not in known_events:
            warnings.append(f"event {seq}: unknown event {event!r} was not checked")

        if event == "ALLOC":
            if not require(
                rec,
                seq,
                "rid",
                "forward_id",
                "req_row",
                "seq_len",
                "full_locs",
                "write_locs",
                "state_slot_virtual",
            ):
                continue
            check_identity_fields(rec, seq)
            full_locs = rec.get("full_locs")
            write_locs = rec.get("write_locs")
            seq_len = integer_field(rec, seq, "seq_len", minimum=0)
            full_locs_int = list_of_ints(full_locs, seq, "full_locs")
            write_locs_int = list_of_ints(write_locs, seq, "write_locs")
            integer_field(rec, seq, "req_row", minimum=1)
            integer_field(rec, seq, "state_slot_virtual", minimum=1)
            if full_locs_int is not None and seq_len is not None:
                if len(full_locs_int) != seq_len:
                    errors.append(
                        f"event {seq}: len(full_locs)={len(full_locs_int)} != seq_len={seq_len}"
                    )
            if full_locs_int is not None and write_locs_int is not None:
                if write_locs_int and full_locs_int[-len(write_locs_int) :] != write_locs_int:
                    errors.append(f"event {seq}: write_locs is not the row suffix")

        elif event == "FORWARD_META":
            if not require(
                rec,
                seq,
                "forward_id",
                "rids",
                "req_rows",
                "state_slots_virtual",
                "state_slots_physical",
                "out_cache_loc",
                "query_start_loc",
            ):
                continue
            if not isinstance(rec.get("forward_id"), str) or not rec.get("forward_id"):
                errors.append(f"event {seq}: forward_id must be a non-empty string")
            rids = rec.get("rids")
            rows = rec.get("req_rows")
            virtual_slots = rec.get("state_slots_virtual")
            physical_slots = rec.get("state_slots_physical")
            out_cache_loc = rec.get("out_cache_loc")
            query_start_loc = rec.get("query_start_loc")
            if not isinstance(rids, list):
                errors.append(f"event {seq}: rids must be a list")
                continue
            if not all(isinstance(rid, str) and rid for rid in rids):
                errors.append(f"event {seq}: rids must contain non-empty strings")
            if not all(isinstance(value, list) for value in (rows, virtual_slots, physical_slots)):
                errors.append(
                    f"event {seq}: req_rows/state_slots_virtual/state_slots_physical must all be lists"
                )
                continue
            if not rids:
                errors.append(f"event {seq}: FORWARD_META must contain at least one request")
            if all(isinstance(rid, str) for rid in rids) and len(rids) != len(set(rids)):
                errors.append(f"event {seq}: rids must be unique within one FORWARD_META")
            if not (len(rids) == len(rows) == len(virtual_slots) == len(physical_slots)):
                errors.append(
                    f"event {seq}: request and slot list lengths differ"
                )
            list_of_ints(rows, seq, "req_rows", minimum=1)
            list_of_ints(virtual_slots, seq, "state_slots_virtual", minimum=1)
            parsed_slots = list_of_ints(physical_slots, seq, "state_slots_physical") or []
            if not isinstance(out_cache_loc, list):
                errors.append(f"event {seq}: out_cache_loc must be a list")
            else:
                list_of_ints(out_cache_loc, seq, "out_cache_loc")
            query = list_of_ints(query_start_loc, seq, "query_start_loc", minimum=0)
            if query is not None:
                if not query or query[0] != 0:
                    errors.append(f"event {seq}: query_start_loc must start at 0")
                if any(right < left for left, right in zip(query, query[1:])):
                    errors.append(f"event {seq}: query_start_loc must be monotonic")
                if len(query) != len(rids) + 1:
                    errors.append(
                        f"event {seq}: query_start_loc length must be len(rids)+1"
                    )
                if query and isinstance(out_cache_loc, list) and query[-1] != len(out_cache_loc):
                    errors.append(
                        f"event {seq}: query_start_loc[-1] must equal len(out_cache_loc)"
                    )
            live = [x for x in parsed_slots if x >= 0]
            if rec.get("require_unique_writers", True) and len(live) != len(set(live)):
                errors.append(f"event {seq}: writable recurrent slot is duplicated")
            if any(x == 0 for x in parsed_slots):
                errors.append(f"event {seq}: slot 0 is the dummy slot, not a request owner")

        elif event == "STATE_BEFORE":
            if not require(
                rec,
                seq,
                "rid",
                "forward_id",
                "layer_id",
                "req_row",
                "state_slot_physical",
                "depth_before",
                "transition_id",
                "input_ids",
            ):
                continue
            check_identity_fields(rec, seq)
            integer_field(rec, seq, "layer_id", minimum=0)
            integer_field(rec, seq, "req_row", minimum=1)
            integer_field(rec, seq, "state_slot_physical", minimum=1)
            tid = str(rec.get("transition_id", ""))
            if not tid:
                errors.append(f"event {seq}: STATE_BEFORE lacks transition_id")
            elif tid in state_before:
                errors.append(f"event {seq}: duplicate STATE_BEFORE {tid}")
            else:
                state_before[tid] = rec
            integer_field(rec, seq, "depth_before", minimum=0)
            if not isinstance(rec.get("input_ids"), list):
                errors.append(f"event {seq}: input_ids must be a list")

        elif event == "STATE_AFTER":
            if not require(
                rec,
                seq,
                "rid",
                "forward_id",
                "layer_id",
                "req_row",
                "state_slot_physical",
                "depth_after",
                "transition_id",
            ):
                continue
            check_identity_fields(rec, seq)
            integer_field(rec, seq, "layer_id", minimum=0)
            integer_field(rec, seq, "req_row", minimum=1)
            integer_field(rec, seq, "state_slot_physical", minimum=1)
            tid = str(rec.get("transition_id", ""))
            before = state_before.pop(tid, None)
            if before is None:
                errors.append(f"event {seq}: STATE_AFTER {tid!r} has no matching BEFORE")
                continue
            for field in ("rid", "forward_id", "layer_id", "req_row", "state_slot_physical"):
                if before.get(field) != rec.get(field):
                    errors.append(f"event {seq}: {tid} changed {field} during one step")
            depth_before = _as_int(before.get("depth_before"))
            depth_after = _as_int(rec.get("depth_after"))
            if depth_after is None or depth_before is None or depth_after < 0:
                errors.append(f"event {seq}: {tid} depth must be integer")
            elif depth_after <= depth_before:
                errors.append(
                    f"event {seq}: {tid} depth did not advance ({depth_before}->{depth_after})"
                )
            before_tensors = snapshot_map(before, seq, f"{tid} BEFORE") or {}
            after_tensors = snapshot_map(rec, seq, f"{tid} AFTER") or {}
            common_names = [name for name in before_tensors if name in after_tensors]
            if not common_names:
                errors.append(f"event {seq}: {tid} has no persistent tensor in both BEFORE and AFTER")
            for name in common_names:
                b = before_tensors[name]
                a = after_tensors[name]
                b_ok = check_snapshot(b, seq, f"{tid}/{name} BEFORE")
                a_ok = check_snapshot(a, seq, f"{tid}/{name} AFTER")
                if not b_ok or not a_ok:
                    continue
                if b.get("data_ptr") != a.get("data_ptr"):
                    errors.append(f"event {seq}: {tid}/{name} data_ptr changed")
                if b.get("shape") != a.get("shape") or b.get("stride") != a.get("stride"):
                    errors.append(f"event {seq}: {tid}/{name} layout changed")
            expect_changed = rec.get("expect_changed", [])
            if not isinstance(expect_changed, list):
                errors.append(f"event {seq}: expect_changed must be a list")
                expect_changed = []
            for name in expect_changed:
                if not isinstance(name, str) or not name:
                    errors.append(f"event {seq}: expect_changed names must be non-empty strings")
                    continue
                if name not in before_tensors or name not in after_tensors:
                    errors.append(f"event {seq}: {tid}/{name} is not present in both tensor snapshots")
                    continue
                old_sig = _signature(before, name)
                new_sig = _signature(rec, name)
                if old_sig is None or new_sig is None:
                    warnings.append(
                        f"event {seq}: cannot value-check {tid}/{name}; rerun with include_values"
                    )
                elif old_sig == new_sig:
                    errors.append(f"event {seq}: {tid}/{name} sampled state did not change")

        elif event == "COW_PLAN":
            if not require(
                rec,
                seq,
                "rid",
                "transition_id",
                "src_slot_physical",
                "dst_slot_physical",
            ):
                continue
            check_identity_fields(rec, seq, include_forward=False)
            tid = rec.get("transition_id")
            if not isinstance(tid, str) or not tid:
                errors.append(f"event {seq}: COW_PLAN lacks transition_id")
                continue
            if tid in cow_plans:
                errors.append(f"event {seq}: duplicate COW_PLAN {tid}")
                continue
            src_slot_int = integer_field(rec, seq, "src_slot_physical", minimum=1)
            dst_slot_int = integer_field(rec, seq, "dst_slot_physical", minimum=1)
            if src_slot_int is None or dst_slot_int is None:
                continue
            elif src_slot_int == dst_slot_int:
                errors.append(f"event {seq}: COW src and dst are the same slot")
            src_owner_state = rec.get("src_owner_state")
            if src_owner_state is not None and str(src_owner_state).upper() != "CACHED":
                errors.append(f"event {seq}: COW source owner must be CACHED")
            if str(src_owner_state).upper() == "CACHED" and not rec.get("src_owner"):
                errors.append(f"event {seq}: CACHED COW source lacks src_owner")
            source_key = (str(rec.get("space", "recurrent")), src_slot_int)
            previous = owner_by_resource.get(source_key)
            if previous is None:
                warnings.append(
                    f"event {seq}: COW source {source_key} has no preceding OWNER=CACHED evidence"
                )
            elif previous[0] != "CACHED":
                errors.append(
                    f"event {seq}: COW source {source_key} owner is {previous[0]}, not CACHED"
                )
            tensors = snapshot_map(rec, seq, f"COW_PLAN {tid}")
            if tensors is None or "src_before" not in tensors:
                errors.append(f"event {seq}: COW_PLAN {tid} needs tensors.src_before")
            elif check_snapshot(tensors["src_before"], seq, f"COW_PLAN {tid}/src_before"):
                cow_plans[tid] = rec

        elif event == "COW_DONE":
            if not require(
                rec,
                seq,
                "rid",
                "transition_id",
                "src_slot_physical",
                "dst_slot_physical",
            ):
                continue
            check_identity_fields(rec, seq, include_forward=False)
            tid = rec.get("transition_id")
            if not isinstance(tid, str) or not tid:
                errors.append(f"event {seq}: COW_DONE lacks transition_id")
                continue
            plan = cow_plans.pop(tid, None)
            if plan is None:
                errors.append(f"event {seq}: COW_DONE {tid!r} has no matching PLAN")
                continue
            for field in ("rid", "src_slot_physical", "dst_slot_physical"):
                if plan.get(field) != rec.get(field):
                    errors.append(f"event {seq}: COW {tid} changed {field}")
            src_slot_int = integer_field(rec, seq, "src_slot_physical", minimum=1)
            dst_slot_int = integer_field(rec, seq, "dst_slot_physical", minimum=1)
            if src_slot_int is not None and dst_slot_int is not None and src_slot_int == dst_slot_int:
                errors.append(f"event {seq}: COW src and dst are the same slot")
            if rec.get("src_owner_state") is not None and str(rec.get("src_owner_state")).upper() != "CACHED":
                errors.append(f"event {seq}: COW source owner must be CACHED")
            if rec.get("src_owner_state") is not None and str(rec.get("src_owner_state")).upper() == "CACHED" and not rec.get("src_owner"):
                errors.append(f"event {seq}: CACHED COW source lacks src_owner")
            tensors = snapshot_map(rec, seq, f"COW_DONE {tid}")
            if tensors is None or "src_after" not in tensors or "dst_after" not in tensors:
                errors.append(f"event {seq}: COW_DONE {tid} needs tensors.src_after and tensors.dst_after")
                continue
            src_after_snapshot = tensors["src_after"]
            dst_after_snapshot = tensors["dst_after"]
            if not check_snapshot(src_after_snapshot, seq, f"COW_DONE {tid}/src_after"):
                continue
            if not check_snapshot(dst_after_snapshot, seq, f"COW_DONE {tid}/dst_after"):
                continue
            src_before_snapshot = plan.get("tensors", {}).get("src_before")
            if isinstance(src_before_snapshot, Mapping):
                if src_before_snapshot.get("data_ptr") != src_after_snapshot.get("data_ptr"):
                    errors.append(f"event {seq}: COW {tid} source data_ptr changed")
                if src_before_snapshot.get("shape") != src_after_snapshot.get("shape") or src_before_snapshot.get("stride") != src_after_snapshot.get("stride"):
                    errors.append(f"event {seq}: COW {tid} source layout changed")
            if src_after_snapshot.get("shape") != dst_after_snapshot.get("shape") or src_after_snapshot.get("stride") != dst_after_snapshot.get("stride"):
                errors.append(f"event {seq}: COW {tid} source/destination layout differs")
            if src_after_snapshot.get("data_ptr") == dst_after_snapshot.get("data_ptr"):
                errors.append(f"event {seq}: COW {tid} source/destination data_ptr aliases")
            if (
                src_after_snapshot.get("storage_ptr") is not None
                and src_after_snapshot.get("storage_ptr") == dst_after_snapshot.get("storage_ptr")
                and src_after_snapshot.get("storage_offset") == dst_after_snapshot.get("storage_offset")
            ):
                errors.append(f"event {seq}: COW {tid} source/destination storage view aliases")
            src_before = _signature(plan, "src_before")
            src_after = _signature(rec, "src_after")
            dst_after = _signature(rec, "dst_after")
            declared_before = rec.get("src_before_sig")
            declared_after = rec.get("src_after_sig")
            declared_dst = rec.get("dst_after_sig")
            if declared_before is not None and src_before is not None and declared_before != src_before:
                errors.append(f"event {seq}: COW {tid} declared src_before_sig disagrees with snapshot")
            if declared_after is not None and src_after is not None and declared_after != src_after:
                errors.append(f"event {seq}: COW {tid} declared src_after_sig disagrees with snapshot")
            if declared_dst is not None and dst_after is not None and declared_dst != dst_after:
                errors.append(f"event {seq}: COW {tid} declared dst_after_sig disagrees with snapshot")
            if src_before is None:
                warnings.append(f"event {seq}: COW {tid} cannot prove source immutability; rerun with include_values")
            elif src_after is not None and src_before != src_after:
                errors.append(f"event {seq}: COW modified the cached source")
            if src_after is None or dst_after is None:
                warnings.append(f"event {seq}: COW {tid} cannot compare source/destination values; rerun with include_values")
            elif src_after != dst_after:
                errors.append(f"event {seq}: COW destination differs from source")

        elif event == "OWNER":
            if not require(rec, seq, "space", "resource_id", "state", "owner", "reason"):
                continue
            state = str(rec.get("state", "")).upper()
            owner = rec.get("owner")
            resource_id = _as_int(rec.get("resource_id"))
            if resource_id is None:
                errors.append(f"event {seq}: resource_id must be an integer")
                continue
            key = (str(rec.get("space")), resource_id)
            if state not in OWNER_STATES:
                errors.append(f"event {seq}: invalid owner state {state}")
                continue
            if state == "FREE" and owner is not None:
                errors.append(f"event {seq}: FREE resource {key} has owner={owner}")
            if state != "FREE" and not owner:
                errors.append(f"event {seq}: {state} resource {key} lacks owner")
            previous = owner_by_resource.get(key)
            if previous is None and state != "FREE":
                warnings.append(
                    f"event {seq}: {key} starts at {state}; include an initial FREE owner event"
                )
            if previous is not None:
                prev_state, prev_owner = previous
                if (prev_state, state) not in OWNER_TRANSITIONS:
                    if not (prev_state == state and prev_owner == owner):
                        errors.append(
                            f"event {seq}: illegal owner transition {key}: "
                            f"{prev_state}/{prev_owner} -> {state}/{owner}"
                        )
            owner_by_resource[key] = (state, owner)

    if seen_seq != list(range(len(seen_seq))):
        errors.append(f"event_seq must be contiguous from 0, got {seen_seq}")
    for tid in sorted(state_before):
        errors.append(f"STATE_BEFORE {tid!r} has no matching AFTER")
    for tid in sorted(cow_plans):
        errors.append(f"COW_PLAN {tid!r} has no matching DONE")
    return ValidationResult(errors=errors, warnings=warnings)


def validate_path(path: str | os.PathLike[str]) -> ValidationResult:
    return validate_records(load_records(path))


def summarize(records: Iterable[Mapping[str, Any]]) -> str:
    header = "seq  event          rid       fwd        layer  row   loc/write        slot  note"
    lines = [header, "-" * len(header)]
    for rec in records:
        locs = rec.get("write_locs", rec.get("out_cache_loc", ""))
        note = rec.get("reason", rec.get("note", ""))
        lines.append(
            f"{str(rec.get('event_seq', '')):>3}  "
            f"{str(rec.get('event', '')):<13.13}  "
            f"{str(rec.get('rid', '')):<8.8}  "
            f"{str(rec.get('forward_id', '')):<9.9}  "
            f"{str(rec.get('layer_id', '')):>5.5}  "
            f"{str(rec.get('req_row', '')):>4.4}  "
            f"{str(locs):<15.15}  "
            f"{str(rec.get('state_slot_physical', rec.get('resource_id', ''))):>4.4}  "
            f"{str(note)}"
        )
    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("summary", "validate"):
        p = sub.add_parser(command)
        p.add_argument("trace", type=Path)
    args = parser.parse_args()
    records = load_records(args.trace)
    if args.command == "summary":
        print(summarize(records))
        return 0
    result = validate_records(records)
    for warning in result.warnings:
        print(f"WARN: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")
    print("PASS" if result.ok else "FAIL")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
