#!/usr/bin/env python3
from bcc import BPF

# eBPF program (tiny) – compiled & loaded by BCC from Python:
bpf_text = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct event {
    u32 pid;
    char comm[16];
    char filename[256];
    int which;   // 0=openat, 1=execve
};

BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(syscalls, sys_enter_openat)
{
    struct event e = {};
    e.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e.comm, sizeof(e.comm));
    e.which = 0;

    // args->filename exists on this tracepoint
    bpf_probe_read_user_str(e.filename, sizeof(e.filename),
                            (const char *)args->filename);

    events.perf_submit(args, &e, sizeof(e));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve)
{
    struct event e = {};
    e.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e.comm, sizeof(e.comm));
    e.which = 1;

    bpf_probe_read_user_str(e.filename, sizeof(e.filename),
                            (const char *)args->filename);

    events.perf_submit(args, &e, sizeof(e));
    return 0;
}
"""

def main():
    b = BPF(text=bpf_text)

    def handle(cpu, data, size):
        ev = b["events"].event(data)
        which = "openat" if ev.which == 0 else "execve"
        # Decode safely (task comm is fixed 16 bytes, may not be fully NUL-terminated)
        comm = ev.comm.decode(errors="ignore").rstrip("\x00")
        fname = ev.filename.decode(errors="replace").rstrip("\x00")
        print(f"[{ev.pid}] {comm:16s} {which:7s} -> {fname}")

    b["events"].open_perf_buffer(handle)
    print("Running… Ctrl-C to stop.")
    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

