# Car worker resume and failure-record audit

The existing GPU6 controller was left running without source changes or a restart. Frozen API v1.1, CarRequest and the shared timing helper also remain byte-for-byte unchanged. Changes are confined to worker entry/record handling, before any formal GPU observations.

Accuracy preserves each flushed accepted pair and now writes its progress JSON and retained-index NPZ atomically. A completed run with matching source/protocol and raw request count can return successfully after a controller interruption. A historical completed result without matching source provenance is kept unchanged. Malformed raw JSONL is reported with the file and line; it is not silently discarded.

Timing resumes committed groups by their source-bound complete marker. Incomplete group files and the old scope GPU telemetry are archived in the next attempt. Python exceptions save a traceback and recover the partially accumulated warmup/formal observations into the failed attempt. Those partial observations are diagnostic and are not counted as a completed formal condition. A final exclusivity check also covers the separate stage profiles before a group is committed.

Memory writes its completed process result atomically, and exceptions retain partial memory rows in the attempt evidence. Each worker type has a process lock, so a child surviving its controller cannot be duplicated by an accidental resume.

Synthetic CPU fault injection passed:40 warmup and4 formal observations were recovered after a deliberately raised exception, the prior telemetry and completed-pair count were preserved, the failed attempt was unchanged after retry, and a duplicate live worker was refused. This is a logging/resume check, not a model or GPU experiment. See attempt_recording_validation.json.

Normal Python failures leave runtime/attempts/<worker>_<timestamp>_<pid>/attempt.json, traceback.txt, prior artifacts and available partial observations. An uncatchable termination can leave a running attempt marker; committed pairs/groups remain the resume boundary. Source changes require a distinct run/provenance path rather than appending to old results.
