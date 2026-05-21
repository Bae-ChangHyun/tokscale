//! Antigravity **CLI** (`agy`) usage adapter.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Child, Command, Stdio};
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
#[cfg(unix)]
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Connect-protocol service exposed by the local Antigravity server.
const SERVICE: &str = "exa.language_server_pb.LanguageServerService";
/// Upper bound on an RPC response body, to bound memory on a hostile peer.
const MAX_RPC_BODY_BYTES: usize = 1024 * 1024;
/// Per-RPC socket timeout.
const RPC_TIMEOUT: Duration = Duration::from_secs(5);
/// How many of the newest `cli-*.log` files to scan for a live port.
const LOG_SCAN_LIMIT: usize = 20;

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().context("Could not determine home directory")
}

/// `~/.gemini/antigravity-cli` — the standalone CLI's per-user data root.
pub fn get_antigravity_cli_dir() -> Result<PathBuf> {
    Ok(home_dir()?.join(".gemini").join("antigravity-cli"))
}

fn get_log_dir() -> Result<PathBuf> {
    Ok(get_antigravity_cli_dir()?.join("log"))
}


/// A snapshot of Antigravity CLI usage, as returned by `GetUserStatus`.
#[derive(Debug, Clone, Serialize)]
pub struct AntigravityCliUsage {
    /// Loopback port the snapshot was read from.
    pub port: u16,
    pub name: Option<String>,
    pub email: Option<String>,
    #[serde(rename = "planName")]
    pub plan_name: Option<String>,
    #[serde(rename = "teamsTier")]
    pub teams_tier: Option<String>,
    #[serde(rename = "promptCredits")]
    pub prompt_credits: CreditBalance,
    #[serde(rename = "flowCredits")]
    pub flow_credits: CreditBalance,
    pub models: Vec<ModelQuota>,
}

/// A credit bucket: how much is left this period vs. the monthly grant.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CreditBalance {
    pub available: Option<i64>,
    pub monthly: Option<i64>,
}

/// Per-model remaining quota, keyed by the human-readable model label.
#[derive(Debug, Clone, Serialize)]
pub struct ModelQuota {
    pub label: String,
    /// Fraction of the model's quota still available, in `0.0..=1.0`.
    #[serde(rename = "remainingFraction")]
    pub remaining_fraction: Option<f64>,
    /// RFC 3339 timestamp at which the quota resets.
    #[serde(rename = "resetTime")]
    pub reset_time: Option<String>,
}


/// Discover the HTTP ports of every reachable `agy` Connect server.
pub fn discover_cli_http_ports() -> Result<Vec<u16>> {
    let mut ports = Vec::new();
    for log in recent_log_files()? {
        if let Some(port) = parse_http_port(&log) {
            if !ports.contains(&port) && probe_healthz(port) {
                ports.push(port);
            }
        }
    }
    Ok(ports)
}

/// The newest [`LOG_SCAN_LIMIT`] `cli-*.log` files, most-recent first.
fn recent_log_files() -> Result<Vec<PathBuf>> {
    let dir = get_log_dir()?;
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<(SystemTime, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        let is_cli_log = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("cli-") && name.ends_with(".log"));
        if !is_cli_log {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .unwrap_or(UNIX_EPOCH);
        entries.push((mtime, path));
    }

    entries.sort_by_key(|(mtime, _)| std::cmp::Reverse(*mtime));
    Ok(entries
        .into_iter()
        .take(LOG_SCAN_LIMIT)
        .map(|(_, path)| path)
        .collect())
}

/// Extract the HTTP port from a single `cli-*.log` file.
fn parse_http_port(log_path: &Path) -> Option<u16> {
    parse_http_port_from(&read_log_capped(log_path)?)
}

/// The HTTP port a single log line announces, if any.
fn http_port_in_line(line: &str) -> Option<u16> {
    let line = line.trim_end();
    if !line.ends_with("for HTTP") {
        return None;
    }
    let idx = line.rfind("at ")?;
    let digits: String = line[idx + 3..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    match digits.parse::<u16>() {
        Ok(port) if port != 0 => Some(port),
        _ => None,
    }
}

/// The process id a single log line announces (`... with pid <N>`), if any.
#[cfg(unix)]
fn pid_in_line(line: &str) -> Option<u32> {
    const MARKER: &str = "with pid ";
    let idx = line.find(MARKER)?;
    let digits: String = line[idx + MARKER.len()..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    digits.parse::<u32>().ok()
}

/// Extract the HTTP port from a `cli-*.log`'s contents (last match wins).
fn parse_http_port_from(contents: &str) -> Option<u16> {
    contents.lines().filter_map(http_port_in_line).next_back()
}

/// `true` if `127.0.0.1:<port>/healthz` answers `200`.
fn probe_healthz(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(RPC_TIMEOUT));
    let _ = stream.set_write_timeout(Some(RPC_TIMEOUT));

    let request =
        format!("GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut status_line = String::new();
    let mut reader = BufReader::new(stream);
    if read_framing_line(&mut reader, &mut status_line).is_err() {
        return false;
    }
    http_status(&status_line) == Some(200)
}


/// Issue a Connect-protocol RPC against the local `agy` server.
fn connect_rpc(port: u16, method: &str, body: &str) -> Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .with_context(|| format!("connecting to agy server on 127.0.0.1:{port}"))?;
    let _ = stream.set_read_timeout(Some(RPC_TIMEOUT));
    let _ = stream.set_write_timeout(Some(RPC_TIMEOUT));

    let request = format!(
        "POST /{SERVICE}/{method} HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Connect-Protocol-Version: 1\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .with_context(|| format!("sending {method} request"))?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    read_framing_line(&mut reader, &mut status_line)?;
    let status = http_status(&status_line).context("malformed HTTP status line")?;

    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    const MAX_HEADERS: usize = 128;
    let mut header_count = 0usize;
    loop {
        let mut header = String::new();
        read_framing_line(&mut reader, &mut header)?;
        let trimmed = header.trim();
        if trimmed.is_empty() {
            break;
        }
        header_count += 1;
        if header_count > MAX_HEADERS {
            anyhow::bail!("agy response sent more than {MAX_HEADERS} headers");
        }
        let lower = trimmed.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            content_length = value.trim().parse::<usize>().ok();
        }
        if let Some(value) = lower.strip_prefix("transfer-encoding:") {
            if value.split(',').any(|coding| coding.trim() == "chunked") {
                chunked = true;
            }
        }
    }

    let body = read_body(&mut reader, content_length, chunked)?;
    if status != 200 {
        anyhow::bail!("agy {method} returned HTTP {status}: {}", body.trim());
    }
    Ok(body)
}

/// Parse the numeric status code out of an HTTP status line.
fn http_status(status_line: &str) -> Option<u16> {
    let mut parts = status_line.split_whitespace();
    if !parts.next()?.starts_with("HTTP/") {
        return None;
    }
    parts.next().and_then(|code| code.parse::<u16>().ok())
}

/// Maximum length of an HTTP/chunked framing line.
const MAX_FRAMING_LINE_BYTES: u64 = 8 * 1024;

/// Read one framing line into `buf`, rejecting an over-long line.
fn read_framing_line<R: BufRead>(reader: &mut R, buf: &mut String) -> Result<usize> {
    let n = reader.take(MAX_FRAMING_LINE_BYTES).read_line(buf)?;
    if n as u64 == MAX_FRAMING_LINE_BYTES && !buf.ends_with('\n') {
        anyhow::bail!("agy response framing line exceeds {MAX_FRAMING_LINE_BYTES} bytes");
    }
    Ok(n)
}

/// Read an HTTP response body, honoring `Content-Length` or chunked framing.
fn read_body<R: BufRead>(
    reader: &mut R,
    content_length: Option<usize>,
    chunked: bool,
) -> Result<String> {
    if chunked {
        let mut out: Vec<u8> = Vec::new();
        loop {
            let mut size_line = String::new();
            read_framing_line(reader, &mut size_line)?;
            let size_hex = size_line.trim().split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_hex, 16)
                .context("malformed chunk size")?;
            if size == 0 {
                break;
            }
            if size > MAX_RPC_BODY_BYTES || out.len() + size > MAX_RPC_BODY_BYTES {
                anyhow::bail!("agy response exceeds {MAX_RPC_BODY_BYTES} bytes");
            }
            let mut chunk = vec![0u8; size];
            reader.read_exact(&mut chunk)?;
            out.extend_from_slice(&chunk);
            let mut crlf = String::new();
            read_framing_line(reader, &mut crlf)?;
            if crlf != "\r\n" && crlf != "\n" {
                anyhow::bail!("malformed chunked framing: expected CRLF after chunk body");
            }
        }
        const MAX_TRAILER_LINES: usize = 64;
        let mut trailer_count = 0usize;
        loop {
            let mut trailer = String::new();
            if read_framing_line(reader, &mut trailer)? == 0 || trailer.trim().is_empty() {
                break;
            }
            trailer_count += 1;
            if trailer_count > MAX_TRAILER_LINES {
                anyhow::bail!("agy response sent more than {MAX_TRAILER_LINES} trailer lines");
            }
        }
        return String::from_utf8(out)
            .context("agy response body was not valid UTF-8 (expected Connect JSON)");
    }

    if let Some(length) = content_length {
        if length > MAX_RPC_BODY_BYTES {
            anyhow::bail!("agy response exceeds {MAX_RPC_BODY_BYTES} bytes");
        }
        let mut bytes = vec![0u8; length];
        reader.read_exact(&mut bytes)?;
        return String::from_utf8(bytes)
            .context("agy response body was not valid UTF-8 (expected Connect JSON)");
    }

    let mut buffer = String::new();
    let read = reader
        .take(MAX_RPC_BODY_BYTES as u64 + 1)
        .read_to_string(&mut buffer)?;
    if read > MAX_RPC_BODY_BYTES {
        anyhow::bail!("agy response exceeds {MAX_RPC_BODY_BYTES} bytes");
    }
    Ok(buffer)
}


/// How long to wait for a freshly spawned `agy` server to answer.
#[cfg(unix)]
const SPAWN_READY_TIMEOUT: Duration = Duration::from_secs(45);
/// How often to re-check while waiting for a spawned server.
#[cfg(unix)]
const SPAWN_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Fetch a usage snapshot, starting a throwaway `agy` server when needed.
pub fn fetch_usage(verbose: bool) -> Result<Option<AntigravityCliUsage>> {
    for port in discover_cli_http_ports().unwrap_or_default() {
        if let Ok(usage) = query_usage(port) {
            return Ok(Some(usage));
        }
    }
    spawn_and_fetch_usage(verbose)
}

/// Query `GetUserStatus` on an already-running server and project the result.
fn query_usage(port: u16) -> Result<AntigravityCliUsage> {
    let raw = connect_rpc(port, "GetUserStatus", "{}")?;
    let json: Value =
        serde_json::from_str(raw.trim()).context("parsing GetUserStatus response")?;
    parse_user_status(port, &json)
}

/// Spawn a short-lived `agy`, read usage from it, then shut it down.
#[cfg(unix)]
fn spawn_and_fetch_usage(verbose: bool) -> Result<Option<AntigravityCliUsage>> {
    if !agy_is_installed() {
        return Ok(None);
    }
    if verbose {
        eprintln!("No running Antigravity CLI server; starting one briefly...");
    }

    let server = spawn_temp_server()?;
    wait_for_usage(server.child.id(), SPAWN_READY_TIMEOUT)
}

#[cfg(not(unix))]
fn spawn_and_fetch_usage(_verbose: bool) -> Result<Option<AntigravityCliUsage>> {
    Ok(None)
}

/// `true` if an executable `agy` is found on `PATH`.
#[cfg(unix)]
fn agy_is_installed() -> bool {
    use std::os::unix::fs::PermissionsExt;
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        std::fs::metadata(dir.join("agy"))
            .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    })
}

/// Process-group id of the live temporary `agy`, or `0` when none is running.
#[cfg(unix)]
static TEMP_AGY_PGID: AtomicI32 = AtomicI32::new(0);

/// Grace period between SIGTERM and SIGKILL when stopping the `agy` group.
#[cfg(unix)]
const GROUP_TERM_GRACE: Duration = Duration::from_millis(300);

/// Kill the temporary `agy` process group (if any), then exit with `exit_code`.
#[cfg(unix)]
fn run_signal_cleanup(exit_code: i32) -> ! {
    let pgid = TEMP_AGY_PGID.load(Ordering::SeqCst);
    if pgid > 0 {
        // SAFETY: `kill(2)` is async-signal-safe.
        unsafe { libc::kill(-pgid, libc::SIGKILL) };
    }
    // SAFETY: `_exit(2)` is async-signal-safe.
    unsafe { libc::_exit(exit_code) }
}

/// `true` if child process `pid` has exited, checked without reaping it.
#[cfg(unix)]
fn child_has_exited(pid: u32) -> bool {
    // SAFETY: a zeroed `siginfo_t` is a valid out-parameter that `waitid` only writes to; `WNOWAIT` leaves the child in its waitable (zombie) state.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
        )
    };
    rc == 0 && info.si_signo != 0
}

/// Terminate `agy`'s whole process group, then reap the direct child.
#[cfg(unix)]
fn terminate_agy_group(child: &mut Child, pid: u32) {
    if pid == 0 {
        let _ = child.kill();
        let _ = child.wait();
        return;
    }
    let group = -(pid as i32);
    // SAFETY: a plain `kill(2)` syscall targeting the child's process group.
    unsafe { libc::kill(group, libc::SIGTERM) };
    let deadline = std::time::Instant::now() + GROUP_TERM_GRACE;
    while std::time::Instant::now() < deadline && !child_has_exited(pid) {
        std::thread::sleep(Duration::from_millis(20));
    }
    // SAFETY: a plain `kill(2)` syscall; SIGKILL the whole group whether or not the leader already exited, so no group member is left behind.
    unsafe { libc::kill(group, libc::SIGKILL) };
    let _ = child.kill();
    let _ = child.wait();
}

/// Block SIGINT/SIGTERM/SIGHUP, returning the previous signal mask.
#[cfg(unix)]
fn block_cleanup_signals() -> libc::sigset_t {
    // SAFETY: standard `sigset`/`sigprocmask` use on owned local sets.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGHUP);
        let mut old: libc::sigset_t = std::mem::zeroed();
        libc::sigprocmask(libc::SIG_BLOCK, &set, &mut old);
        old
    }
}

/// Restore a signal mask saved by [`block_cleanup_signals`].
#[cfg(unix)]
fn restore_signal_mask(mask: &libc::sigset_t) {
    // SAFETY: `mask` came from a prior `sigprocmask`; restoring it is sound.
    unsafe {
        libc::sigprocmask(libc::SIG_SETMASK, mask, std::ptr::null_mut());
    }
}

/// RAII guard that blocks SIGINT/SIGTERM/SIGHUP for its lifetime.
#[cfg(unix)]
struct SignalMaskGuard {
    saved: libc::sigset_t,
}

#[cfg(unix)]
impl SignalMaskGuard {
    fn block() -> Self {
        Self {
            saved: block_cleanup_signals(),
        }
    }
}

#[cfg(unix)]
impl Drop for SignalMaskGuard {
    fn drop(&mut self) {
        restore_signal_mask(&self.saved);
    }
}

/// Read a `cli-*.log` into a string, bounded to a sane size.
fn read_log_capped(path: &Path) -> Option<String> {
    const MAX_LOG_BYTES: u64 = 512 * 1024;
    let file = fs::File::open(path).ok()?;
    let mut buf = String::new();
    file.take(MAX_LOG_BYTES).read_to_string(&mut buf).ok()?;
    Some(buf)
}

/// Set `FD_CLOEXEC` on a raw fd.
#[cfg(unix)]
fn set_cloexec(fd: libc::c_int) -> std::io::Result<()> {
    // SAFETY: fcntl with these commands has no memory-safety obligations.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Set `O_NONBLOCK` on a raw fd.
#[cfg(unix)]
fn set_nonblocking(fd: libc::c_int) -> std::io::Result<()> {
    // SAFETY: fcntl with these commands has no memory-safety obligations.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Spawn a temporary `agy` server attached to a pseudo-terminal.
#[cfg(unix)]
fn spawn_temp_server() -> Result<TempAgyServer> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::process::CommandExt;

    let _signal_fence = SignalMaskGuard::block();

    let mut master_fd: libc::c_int = -1;
    let mut slave_fd: libc::c_int = -1;
    let mut winsize = libc::winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: on success openpty writes two fresh, owned fds into master_fd and slave_fd. The name pointer is null because we do not need the path.
    let rc = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null_mut::<libc::termios>(),
            &mut winsize as *mut libc::winsize,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error())
            .context("openpty failed while preparing a temporary agy server");
    }
    if let Err(err) = set_cloexec(master_fd)
        .and_then(|()| set_cloexec(slave_fd))
        .and_then(|()| set_nonblocking(master_fd))
    {
        // SAFETY: closing the two fds we own; nothing else references them yet.
        unsafe {
            libc::close(master_fd);
            libc::close(slave_fd);
        }
        return Err(err).context("configuring temporary agy PTY file descriptors");
    }
    // SAFETY: both fds were just created by openpty and are owned by us.
    let master = unsafe { OwnedFd::from_raw_fd(master_fd) };
    let slave = unsafe { OwnedFd::from_raw_fd(slave_fd) };

    let slave_stdin = slave.try_clone()?;
    let slave_stdout = slave.try_clone()?;
    let slave_stderr = slave.try_clone()?;
    let slave_ctty = slave.try_clone()?;

    let mut command = Command::new("agy");
    command
        .stdin(Stdio::from(slave_stdin))
        .stdout(Stdio::from(slave_stdout))
        .stderr(Stdio::from(slave_stderr));

    // SAFETY: `pre_exec` runs in the forked child before `exec`; `setsid` and `ioctl` are async-signal-safe. This detaches the child into a new session and makes the PTY its controlling terminal — required by `agy`'s TUI layer — and, as the session leader, the child is its own process-group leader so the negative-PID kill on drop cannot reach tokscale itself.
    unsafe {
        command.pre_exec(move || {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::ioctl(slave_ctty.as_raw_fd(), libc::TIOCSCTTY as _, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let mut child = command
        .spawn()
        .context("spawning `agy` (is the Antigravity CLI installed?)")?;
    drop(slave); // the child holds its own dups now

    let drain_stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&drain_stop);
    let drain = std::thread::Builder::new()
        .name("agy-pty-drain".to_string())
        .spawn(move || {
            let fd = master.as_raw_fd();
            let mut sink = [0u8; 8192];
            loop {
                // SAFETY: `master` owns `fd` for the whole lifetime of this loop.
                let n = unsafe {
                    libc::read(fd, sink.as_mut_ptr() as *mut libc::c_void, sink.len())
                };
                if n > 0 {
                    continue;
                }
                if n == 0 {
                    break; // EOF: the slave end is fully closed
                }
                if thread_stop.load(Ordering::Relaxed) {
                    break;
                }
                match std::io::Error::last_os_error().raw_os_error() {
                    Some(libc::EINTR) => {} // interrupted — retry the read
                    Some(code) if code == libc::EAGAIN || code == libc::EWOULDBLOCK => {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    _ => break, // hard error / PTY torn down
                }
            }
            drop(master);
        });
    let drain = match drain {
        Ok(handle) => handle,
        Err(err) => {
            let pid = child.id();
            terminate_agy_group(&mut child, pid);
            return Err(err).context("spawning the agy PTY drain thread");
        }
    };

    TEMP_AGY_PGID.store(child.id() as i32, Ordering::SeqCst);
    let mut sig_ids = Vec::new();
    for (signal, exit_code) in [
        (libc::SIGINT, 130),
        (libc::SIGTERM, 143),
        (libc::SIGHUP, 129),
    ] {
        // SAFETY: signal-hook invokes this action from its async-signal-safe handler trampoline; the action calls only `kill` and `_exit`, which are themselves async-signal-safe.
        if let Ok(id) = unsafe {
            signal_hook::low_level::register(signal, move || run_signal_cleanup(exit_code))
        } {
            sig_ids.push(id);
        }
    }

    Ok(TempAgyServer {
        child,
        drain: Some(drain),
        drain_stop,
        sig_ids,
    })
}

/// Poll the spawned `agy` until it answers `GetUserStatus`, or time out.
#[cfg(unix)]
fn wait_for_usage(pid: u32, timeout: Duration) -> Result<Option<AntigravityCliUsage>> {
    let deadline = std::time::Instant::now() + timeout;
    let mut last_error: Option<anyhow::Error> = None;
    loop {
        if let Ok(Some(port)) = http_port_for_pid(pid) {
            match query_usage(port) {
                Ok(usage) => return Ok(Some(usage)),
                Err(err) => last_error = Some(err),
            }
        }
        if child_has_exited(pid) {
            return match last_error {
                Some(err) => {
                    Err(err.context("agy exited before returning a usable response"))
                }
                None => anyhow::bail!("agy exited before its server was reachable"),
            };
        }
        if std::time::Instant::now() >= deadline {
            return match last_error {
                Some(err) => Err(err
                    .context("agy server started but never returned a usable response")),
                None => Ok(None),
            };
        }
        std::thread::sleep(SPAWN_POLL_INTERVAL);
    }
}

/// HTTP port of the running `agy` whose log records process id `pid`, when that
#[cfg(unix)]
fn http_port_for_pid(pid: u32) -> Result<Option<u16>> {
    for log in recent_log_files()? {
        let Some(contents) = read_log_capped(&log) else {
            continue;
        };
        if let Some(port) = http_port_for_pid_in(&contents, pid) {
            if probe_healthz(port) {
                return Ok(Some(port));
            }
        }
    }
    Ok(None)
}

/// The last HTTP port logged in `contents` while process `pid` was the active
#[cfg(unix)]
fn http_port_for_pid_in(contents: &str, pid: u32) -> Option<u16> {
    let mut current_pid = None;
    let mut port = None;
    for line in contents.lines() {
        if let Some(found) = pid_in_line(line) {
            current_pid = Some(found);
        }
        if current_pid == Some(pid) {
            if let Some(found) = http_port_in_line(line) {
                port = Some(found);
            }
        }
    }
    port
}

/// RAII guard for a temporary `agy` server — killed when dropped.
#[cfg(unix)]
struct TempAgyServer {
    /// The `agy` process; its PID doubles as the process-group id.
    child: Child,
    /// The PTY-master drain thread, joined in `Drop`.
    drain: Option<std::thread::JoinHandle<()>>,
    /// Signals the drain thread to stop, bounding the `Drop` join.
    drain_stop: Arc<AtomicBool>,
    /// Scoped SIGINT/SIGTERM/SIGHUP handler ids, unregistered on `Drop`.
    sig_ids: Vec<signal_hook::SigId>,
}

#[cfg(unix)]
impl Drop for TempAgyServer {
    fn drop(&mut self) {
        let _signal_fence = SignalMaskGuard::block();
        let pid = self.child.id();
        terminate_agy_group(&mut self.child, pid);
        TEMP_AGY_PGID.store(0, Ordering::SeqCst);
        for id in self.sig_ids.drain(..) {
            signal_hook::low_level::unregister(id);
        }
        self.drain_stop.store(true, Ordering::Relaxed);
        if let Some(drain) = self.drain.take() {
            let _ = drain.join();
        }
    }
}

/// Project a `GetUserStatus` JSON response into [`AntigravityCliUsage`].
fn parse_user_status(port: u16, json: &Value) -> Result<AntigravityCliUsage> {
    let status = match json.get("userStatus").filter(|value| value.is_object()) {
        Some(status) => status,
        None => {
            if let Some(message) = json.get("message").and_then(Value::as_str) {
                let code = json
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or("error");
                anyhow::bail!("agy GetUserStatus failed ({code}): {message}");
            }
            anyhow::bail!("GetUserStatus response has no `userStatus` object");
        }
    };
    let null = Value::Null;
    let plan_status = status.get("planStatus").unwrap_or(&null);
    let plan_info = plan_status.get("planInfo").unwrap_or(&null);

    let models = status
        .pointer("/cascadeModelConfigData/clientModelConfigs")
        .and_then(Value::as_array)
        .map(|configs| configs.iter().filter_map(parse_model_quota).collect())
        .unwrap_or_default();

    Ok(AntigravityCliUsage {
        port,
        name: string_field(status, "name"),
        email: string_field(status, "email"),
        plan_name: string_field(plan_info, "planName"),
        teams_tier: string_field(plan_info, "teamsTier"),
        prompt_credits: CreditBalance {
            available: plan_status.get("availablePromptCredits").and_then(as_i64),
            monthly: plan_info.get("monthlyPromptCredits").and_then(as_i64),
        },
        flow_credits: CreditBalance {
            available: plan_status.get("availableFlowCredits").and_then(as_i64),
            monthly: plan_info.get("monthlyFlowCredits").and_then(as_i64),
        },
        models,
    })
}

fn parse_model_quota(config: &Value) -> Option<ModelQuota> {
    let label = config.get("label")?.as_str()?.to_string();
    let quota = config.get("quotaInfo");
    Some(ModelQuota {
        label,
        remaining_fraction: quota
            .and_then(|q| q.get("remainingFraction"))
            .and_then(as_f64),
        reset_time: quota
            .and_then(|q| q.get("resetTime"))
            .and_then(Value::as_str)
            .map(String::from),
    })
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(String::from)
}

/// proto-JSON renders `int32` as a number but `int64` as a quoted string.
fn as_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}

fn as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}


/// Implements `tokscale antigravity usage [--json]`.
pub fn run_antigravity_cli_usage(json: bool) -> Result<()> {
    match fetch_usage(!json)? {
        None => {
            if json {
                println!("null");
            } else {
                eprintln!(
                    "Could not read Antigravity CLI usage.\n\
                     Ensure the `agy` CLI is installed and signed in \
                     (run `agy` once to sign in)."
                );
            }
            Ok(())
        }
        Some(usage) if json => {
            println!("{}", serde_json::to_string_pretty(&usage)?);
            Ok(())
        }
        Some(usage) => {
            print_usage(&usage);
            Ok(())
        }
    }
}

fn print_usage(usage: &AntigravityCliUsage) {
    use colored::Colorize;
    println!("{}", "Antigravity CLI usage".bold());
    print_usage_body(usage);
}

/// Print account, plan, credit balances, and per-model quota lines.
fn print_usage_body(usage: &AntigravityCliUsage) {
    use colored::Colorize;

    let account = match (&usage.name, &usage.email) {
        (Some(name), Some(email)) => format!("{name} <{email}>"),
        (Some(name), None) => name.clone(),
        (None, Some(email)) => email.clone(),
        (None, None) => "(unknown account)".to_string(),
    };
    let plan = match (&usage.plan_name, &usage.teams_tier) {
        (Some(name), Some(tier)) => format!("{name} ({tier})"),
        (Some(name), None) => name.clone(),
        (None, Some(tier)) => tier.clone(),
        (None, None) => "(unknown plan)".to_string(),
    };

    println!("  {} {account}", "account:".bright_black());
    println!("  {} {plan}", "plan:   ".bright_black());
    println!(
        "  {} {}",
        "prompt: ".bright_black(),
        format_credit(&usage.prompt_credits)
    );
    println!(
        "  {} {}",
        "flow:   ".bright_black(),
        format_credit(&usage.flow_credits)
    );

    if !usage.models.is_empty() {
        println!("  {}", "models:".bright_black());
        for model in &usage.models {
            let remaining = match model.remaining_fraction {
                Some(fraction) => format!("{:>3.0}%", fraction.clamp(0.0, 1.0) * 100.0),
                None => "  ?".to_string(),
            };
            let reset = model.reset_time.as_deref().unwrap_or("-");
            println!(
                "    - {:<34} {} left  resets {}",
                model.label,
                remaining,
                reset.bright_black()
            );
        }
    }
}

fn format_credit(credit: &CreditBalance) -> String {
    match (credit.available, credit.monthly) {
        (Some(available), Some(monthly)) => {
            format!("{available} available / {monthly} monthly")
        }
        (Some(available), None) => format!("{available} available"),
        (None, Some(monthly)) => format!("{monthly} monthly"),
        (None, None) => "(unavailable)".to_string(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_http_port_picks_http_not_grpc() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("cli-20260520_140756.log");
        let mut file = fs::File::create(&log).unwrap();
        writeln!(
            file,
            "I0520 14:07:56.636760 server.go:485] Language server listening on random port at 38399 for HTTPS (gRPC)"
        )
        .unwrap();
        writeln!(
            file,
            "I0520 14:07:56.636782 server.go:492] Language server listening on random port at 34163 for HTTP"
        )
        .unwrap();

        assert_eq!(parse_http_port(&log), Some(34163));
    }

    #[test]
    fn parse_http_port_absent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("cli-empty.log");
        fs::write(&log, "I0520 nothing relevant here\n").unwrap();
        assert_eq!(parse_http_port(&log), None);
    }

    #[test]
    fn as_i64_accepts_number_and_string() {
        assert_eq!(as_i64(&serde_json::json!(50000)), Some(50000));
        assert_eq!(as_i64(&serde_json::json!("16384")), Some(16384));
        assert_eq!(as_i64(&serde_json::json!(true)), None);
    }

    #[test]
    fn parse_user_status_extracts_credits_and_models() {
        let json = serde_json::json!({
            "userStatus": {
                "name": "Test User",
                "email": "test@example.com",
                "planStatus": {
                    "planInfo": {
                        "planName": "Pro",
                        "teamsTier": "TEAMS_TIER_PRO",
                        "monthlyPromptCredits": 50000,
                        "monthlyFlowCredits": 150000
                    },
                    "availablePromptCredits": 500,
                    "availableFlowCredits": 100
                },
                "cascadeModelConfigData": {
                    "clientModelConfigs": [
                        {
                            "label": "Gemini 3.1 Pro (High)",
                            "quotaInfo": {
                                "remainingFraction": 1,
                                "resetTime": "2026-05-20T10:19:58Z"
                            }
                        },
                        { "label": "no-quota-model" }
                    ]
                }
            }
        });

        let usage = parse_user_status(34163, &json).unwrap();
        assert_eq!(usage.plan_name.as_deref(), Some("Pro"));
        assert_eq!(usage.prompt_credits.available, Some(500));
        assert_eq!(usage.prompt_credits.monthly, Some(50000));
        assert_eq!(usage.flow_credits.available, Some(100));
        assert_eq!(usage.models.len(), 2);
        assert_eq!(usage.models[0].label, "Gemini 3.1 Pro (High)");
        assert_eq!(usage.models[0].remaining_fraction, Some(1.0));
        assert_eq!(usage.models[1].remaining_fraction, None);
    }

    #[test]
    fn http_status_parses_valid_and_rejects_non_http() {
        assert_eq!(http_status("HTTP/1.1 200 OK"), Some(200));
        assert_eq!(http_status("HTTP/1.1 404"), Some(404));
        assert_eq!(http_status("GARBAGE 200 OK"), None);
        assert_eq!(http_status(""), None);
    }

    #[test]
    fn parse_http_port_rejects_zero() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("cli-zero.log");
        fs::write(&log, "listening on random port at 0 for HTTP\n").unwrap();
        assert_eq!(parse_http_port(&log), None);
    }

    #[cfg(unix)]
    #[test]
    fn pid_in_line_reads_pid_marker() {
        let line =
            "I0520 14:07:56 server.go:1295] Starting language server process with pid 122021";
        assert_eq!(pid_in_line(line), Some(122021));
        assert_eq!(pid_in_line("nothing here"), None);
    }

    #[cfg(unix)]
    #[test]
    fn http_port_for_pid_in_pairs_port_with_its_own_pid() {
        let log = "started with pid 100\n\
                   listening on random port at 4000 for HTTP\n\
                   restarted with pid 200\n\
                   listening on random port at 5000 for HTTP\n";
        assert_eq!(http_port_for_pid_in(log, 100), Some(4000));
        assert_eq!(http_port_for_pid_in(log, 200), Some(5000));
        assert_eq!(http_port_for_pid_in(log, 999), None);
    }

    #[test]
    fn read_body_content_length() {
        let mut reader = std::io::Cursor::new(b"hello world".to_vec());
        assert_eq!(
            read_body(&mut reader, Some(11), false).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn read_body_chunked_reassembles() {
        let raw = "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let mut reader = std::io::Cursor::new(raw.as_bytes().to_vec());
        assert_eq!(read_body(&mut reader, None, true).unwrap(), "hello world");
    }

    #[test]
    fn read_body_chunked_strips_extension() {
        let raw = "5;name=value\r\nhello\r\n0\r\n\r\n";
        let mut reader = std::io::Cursor::new(raw.as_bytes().to_vec());
        assert_eq!(read_body(&mut reader, None, true).unwrap(), "hello");
    }

    #[test]
    fn read_body_rejects_oversized_content_length() {
        let mut reader = std::io::Cursor::new(vec![b'x'; 8]);
        assert!(read_body(&mut reader, Some(MAX_RPC_BODY_BYTES + 1), false).is_err());
    }

    #[test]
    fn read_body_rejects_oversized_chunk() {
        let raw = format!("{:x}\r\n", MAX_RPC_BODY_BYTES + 1);
        let mut reader = std::io::Cursor::new(raw.into_bytes());
        assert!(read_body(&mut reader, None, true).is_err());
    }

    #[test]
    fn read_body_close_delimited() {
        let mut reader = std::io::Cursor::new(b"plain body".to_vec());
        assert_eq!(read_body(&mut reader, None, false).unwrap(), "plain body");
    }

    #[test]
    fn read_body_rejects_oversized_close_delimited() {
        let mut reader = std::io::Cursor::new(vec![b'x'; MAX_RPC_BODY_BYTES + 1]);
        assert!(read_body(&mut reader, None, false).is_err());
    }

    #[test]
    fn read_body_rejects_malformed_chunk_crlf() {
        let raw = "5\r\nhelloXX0\r\n\r\n";
        let mut reader = std::io::Cursor::new(raw.as_bytes().to_vec());
        assert!(read_body(&mut reader, None, true).is_err());
    }

    #[test]
    fn parse_user_status_rejects_missing_user_status() {
        let json = serde_json::json!({ "somethingElse": {} });
        assert!(parse_user_status(34163, &json).is_err());
    }
}
