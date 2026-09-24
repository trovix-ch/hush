//! Hold Right Ctrl in a text field and release to insert a numbered line; Tray, Quit exits.
//! `--simulate-hotkey` drives the same flow into a scratch Notepad window and reads it back.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use hush_core::cancel::CancelToken;
use hush_core::insert::{InsertPolicy, Inserter, StrategyChain};
use hush_platform_windows::clipboard::{ClipboardSnapshot, DEFAULT_SNAPSHOT_CAP};
use hush_platform_windows::focus::WinFocus;
use hush_platform_windows::hook::{HotkeyConfig, HotkeyEvent, HotkeyHook};
use hush_platform_windows::input::WinInput;
use hush_platform_windows::overlay::OverlayState;
use hush_platform_windows::sound::{self, Cue};
use hush_platform_windows::tray::TrayEvent;
use hush_platform_windows::ui_thread::{self, UiHandle, UiOptions};

use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumChildWindows, EnumWindows, GetClassNameW, GetWindowTextW, IsWindowVisible,
    SMTO_ABORTIFHUNG, SendMessageTimeoutW, WM_GETTEXT, WM_GETTEXTLENGTH,
};
use windows::core::BOOL;

enum Ev {
    Hotkey(HotkeyEvent),
    Tray(TrayEvent),
}

struct Args {
    simulate: bool,
    runs: u32,
    hold_ms: u64,
    key: String,
    tray: bool,
    type_test: bool,
}

fn args() -> Args {
    let mut a = Args {
        simulate: false,
        runs: 3,
        hold_ms: 400,
        key: "RightCtrl".into(),
        tray: true,
        type_test: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--simulate-hotkey" => a.simulate = true,
            "--runs" => a.runs = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
            "--hold-ms" => a.hold_ms = it.next().and_then(|v| v.parse().ok()).unwrap_or(400),
            "--key" => a.key = it.next().unwrap_or(a.key),
            "--no-tray" => a.tray = false,
            "--type-test" => a.type_test = true,
            other => {
                eprintln!(
                    "unknown argument {other}\nplatform-demo [--simulate-hotkey] [--runs N] [--hold-ms MS] [--key NAME] [--no-tray] [--type-test]"
                );
                std::process::exit(2);
            }
        }
    }
    a
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,hush_platform_windows=debug".into()),
        )
        .init();
    let args = args();

    let _instance = match ui_thread::acquire_single_instance(ui_thread::INSTANCE_MUTEX) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    if let Some(p) = ui_thread::AppPaths::resolve() {
        println!(
            "paths: config {} | models {}",
            p.config_file.display(),
            p.models_dir.display()
        );
    }

    let (ui, tray_rx) = UiHandle::start(UiOptions {
        tray: args.tray,
        ..Default::default()
    })
    .expect("UI thread");
    let (ev_tx, ev_rx) = mpsc::channel::<Ev>();

    let (hk_tx, hk_rx) = mpsc::sync_channel::<HotkeyEvent>(64);
    let config = HotkeyConfig::parse(&args.key).expect("hotkey");
    let hook = HotkeyHook::install(config, hk_tx.clone()).expect("hook");
    {
        let tx = ev_tx.clone();
        std::thread::spawn(move || {
            while let Ok(e) = hk_rx.recv() {
                if tx.send(Ev::Hotkey(e)).is_err() {
                    break;
                }
            }
        });
        let tx = ev_tx.clone();
        std::thread::spawn(move || {
            while let Ok(e) = tray_rx.recv() {
                if tx.send(Ev::Tray(e)).is_err() {
                    break;
                }
            }
        });
    }

    let clipboard = ui.clipboard().clone();
    let input = WinInput::new();
    let focus = WinFocus::new();
    let mut chain = StrategyChain::new(
        clipboard.clone(),
        input.clone(),
        focus.clone(),
        InsertPolicy::default(),
    );

    let mut notepad: Option<(HWND, PathBuf)> = None;
    if args.simulate {
        let (hwnd, file) = open_notepad();
        println!("notepad window {:?} on {}", hwnd.0, file.display());
        notepad = Some((hwnd, file));
        let hk = hk_tx.clone();
        let runs = args.runs;
        let hold = args.hold_ms;
        let target = hwnd.0 as isize;
        std::thread::spawn(move || {
            for _ in 0..runs {
                let ok = hush_platform_windows::focus::refocus_raw(target);
                println!("[sim] notepad foreground: {ok}");
                std::thread::sleep(Duration::from_millis(250));
                let _ = hk.send(HotkeyEvent::Down { at: Instant::now() });
                std::thread::sleep(Duration::from_millis(hold));
                let _ = hk.send(HotkeyEvent::Up { at: Instant::now() });
                // Room for the ~1 s third-party restore delay plus a read-back.
                std::thread::sleep(Duration::from_millis(2200));
            }
        });
    } else {
        println!(
            "hold {} in a text field and release; tray -> Quit to exit",
            args.key
        );
    }

    let mut n = 0u32;
    let mut captured = None;
    let mut paused = false;
    let mut delivered = 0u32;
    loop {
        let Ok(ev) = ev_rx.recv_timeout(Duration::from_millis(200)) else {
            if args.simulate && n >= args.runs && captured.is_none() {
                break;
            }
            continue;
        };
        match ev {
            Ev::Hotkey(HotkeyEvent::Down { at }) if !paused => {
                let t = Instant::now();
                let snap = focus.capture();
                println!(
                    "\n[down] +{:.1} ms after key; focus {:?} {:?} elevated={} password={} remote={} (capture {:.1} ms)",
                    (t - at).as_secs_f64() * 1e3,
                    snap.exe,
                    snap.title,
                    snap.elevated,
                    snap.is_password,
                    snap.remote_session,
                    t.elapsed().as_secs_f64() * 1e3
                );
                sound::play(Cue::Start);
                ui.set_overlay(OverlayState::Listening { level: 0.6 });
                captured = Some(snap);
            }
            Ev::Hotkey(HotkeyEvent::Up { at }) => {
                let Some(snap) = captured.take() else {
                    continue;
                };
                sound::play(Cue::Stop);
                ui.set_overlay(OverlayState::Inserting);
                n += 1;
                let text = format!("hello from hush {n}");
                let before = clipboard.snapshot(DEFAULT_SNAPSHOT_CAP).ok();
                let t0 = Instant::now();
                let outcome = chain.insert(&snap.to_core(), &text, &CancelToken::new());
                let took = t0.elapsed();
                let render = clipboard.first_render();
                let chord = input.last_chord_at();
                println!(
                    "[up]   +{:.1} ms after key; outcome {:?} in {:.1} ms",
                    (t0 - at).as_secs_f64() * 1e3,
                    outcome,
                    took.as_secs_f64() * 1e3
                );
                match (&render, chord) {
                    (Some(r), Some(c)) => println!(
                        "       first reader {:?} (pid {:?}) at {:+.1} ms relative to the chord; renders {}",
                        r.reader_exe,
                        r.reader_pid,
                        r.offset_ms(c),
                        clipboard.render_count()
                    ),
                    (Some(r), None) => {
                        println!("       first reader {:?}, no chord sent", r.reader_exe)
                    }
                    (None, _) => println!("       nobody read the clipboard"),
                }
                match &outcome {
                    Ok(o) if o.delivered() => {
                        delivered += 1;
                        ui.set_overlay(OverlayState::Done { message: None });
                    }
                    Ok(o) => {
                        sound::play(Cue::Error);
                        ui.set_overlay(OverlayState::Error {
                            message: o.message().unwrap_or_default(),
                        });
                    }
                    Err(e) => {
                        sound::play(Cue::Error);
                        ui.set_overlay(OverlayState::Error {
                            message: e.to_string(),
                        });
                    }
                }
                if let Some(before) = before {
                    let clip = clipboard.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(1400));
                        let after = clip.snapshot(DEFAULT_SNAPSHOT_CAP).ok();
                        report_restore(&before, after.as_ref());
                    });
                }
                if let Some((hwnd, _)) = &notepad {
                    std::thread::sleep(Duration::from_millis(150));
                    let found = notepad_text(*hwnd)
                        .iter()
                        .any(|t| t.lines().any(|l| l.contains(&text)));
                    println!("       notepad read-back contains {text:?}: {found}");
                }
            }
            Ev::Hotkey(HotkeyEvent::Cancel { .. }) => {
                captured = None;
                sound::play(Cue::Cancel);
                ui.set_overlay(OverlayState::Hidden);
                println!("[cancel]");
            }
            Ev::Hotkey(HotkeyEvent::HookReinstalled { reason, .. }) => {
                println!("[hook reinstalled: {reason:?}]");
            }
            Ev::Hotkey(_) => {}
            Ev::Tray(TrayEvent::Quit) => break,
            Ev::Tray(TrayEvent::TogglePause) => {
                paused = !paused;
                ui.set_paused(paused);
                println!("[tray] paused={paused}");
            }
            Ev::Tray(TrayEvent::About) => ui.show_about(),
            Ev::Tray(TrayEvent::OpenConfig) => {
                if let Some(p) = ui_thread::AppPaths::resolve() {
                    let _ = ui_thread::open_path(&p.config_file);
                }
            }
            Ev::Tray(other) => println!("[tray] {other:?} (no history in the demo)"),
        }
    }
    if args.simulate {
        std::thread::sleep(Duration::from_millis(1600));
        println!("\nsummary: {delivered}/{n} delivered");
        if let Some((hwnd, _)) = &notepad {
            if args.type_test {
                let text = "\ntyped: grüße, naïve 😀 done\nline two";
                let ok = hush_platform_windows::focus::refocus_raw(hwnd.0 as isize);
                std::thread::sleep(Duration::from_millis(200));
                let r = input.type_text(text);
                std::thread::sleep(Duration::from_millis(600));
                let landed = notepad_text(*hwnd)
                    .iter()
                    .any(|t| t.replace("\r\n", "\n").replace('\r', "\n").contains(text));
                println!("\n[type] foreground={ok} result {r:?}; landed intact: {landed}");
            }
            for t in notepad_text(*hwnd) {
                println!("notepad text: {t:?}");
            }
        }
    }
    drop(hook);
    ui.shutdown();
}

fn report_restore(before: &ClipboardSnapshot, after: Option<&ClipboardSnapshot>) {
    let Some(after) = after else {
        println!("       restore: could not read the clipboard afterwards");
        return;
    };
    let same_text = before.text() == after.text();
    let same_bytes = before.formats.iter().all(|b| {
        after
            .formats
            .iter()
            .any(|a| a.format == b.format && a.bytes == b.bytes)
    });
    println!(
        "       restore: text identical={same_text} ({:?}); formats before {} after {}; every format byte-identical={same_bytes}",
        after.text().map(|t| t.chars().take(40).collect::<String>()),
        before.formats.len(),
        after.formats.len()
    );
}

fn open_notepad() -> (HWND, PathBuf) {
    let file = std::env::temp_dir().join(format!("hush-demo-{}.txt", std::process::id()));
    std::fs::write(&file, "").expect("scratch file");
    let name = file.file_name().unwrap().to_string_lossy().into_owned();
    // An inherited stdout pipe would stay open for as long as Notepad lives.
    let mut child = std::process::Command::new("notepad.exe")
        .arg(&file)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("notepad");
    std::thread::spawn(move || child.wait());
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Some(h) = find_window_titled(&name) {
            return (h, file);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("no Notepad window titled {name}");
}

fn find_window_titled(needle: &str) -> Option<HWND> {
    struct Ctx<'a> {
        needle: &'a str,
        found: Option<HWND>,
    }
    unsafe extern "system" fn cb(hwnd: HWND, lp: LPARAM) -> BOOL {
        // SAFETY: `lp` is the &mut Ctx passed below, alive for the enumeration.
        let ctx = unsafe { &mut *(lp.0 as *mut Ctx) };
        // SAFETY: plain FFI queries on an enumerated window.
        if !unsafe { IsWindowVisible(hwnd) }.as_bool() {
            return BOOL(1);
        }
        let mut buf = [0u16; 512];
        // SAFETY: as above.
        let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
        if String::from_utf16_lossy(&buf[..n.max(0) as usize]).contains(ctx.needle) {
            ctx.found = Some(hwnd);
            return BOOL(0);
        }
        BOOL(1)
    }
    let mut ctx = Ctx {
        needle,
        found: None,
    };
    // SAFETY: the callback only touches `ctx` through the pointer we pass.
    let _ = unsafe { EnumWindows(Some(cb), LPARAM(&mut ctx as *mut _ as isize)) };
    ctx.found
}

fn notepad_text(top: HWND) -> Vec<String> {
    unsafe extern "system" fn cb(hwnd: HWND, lp: LPARAM) -> BOOL {
        // SAFETY: `lp` is the &mut Vec passed below.
        let out = unsafe { &mut *(lp.0 as *mut Vec<String>) };
        let mut cls = [0u16; 128];
        // SAFETY: plain FFI query.
        let n = unsafe { GetClassNameW(hwnd, &mut cls) };
        let class = String::from_utf16_lossy(&cls[..n.max(0) as usize]);
        if !class.to_lowercase().contains("edit") {
            return BOOL(1);
        }
        let mut len = 0usize;
        // SAFETY: cross-process WM_GETTEXT with a timeout; the buffer outlives the call.
        unsafe {
            SendMessageTimeoutW(
                hwnd,
                WM_GETTEXTLENGTH,
                WPARAM(0),
                LPARAM(0),
                SMTO_ABORTIFHUNG,
                1000,
                Some(&mut len),
            );
            let mut buf = vec![0u16; len + 1];
            let mut got = 0usize;
            SendMessageTimeoutW(
                hwnd,
                WM_GETTEXT,
                WPARAM(buf.len()),
                LPARAM(buf.as_mut_ptr() as isize),
                SMTO_ABORTIFHUNG,
                1000,
                Some(&mut got),
            );
            out.push(String::from_utf16_lossy(&buf[..got.min(len)]));
        }
        BOOL(1)
    }
    let mut out = Vec::new();
    // SAFETY: the callback only touches `out` through the pointer we pass.
    let _ = unsafe { EnumChildWindows(Some(top), Some(cb), LPARAM(&mut out as *mut _ as isize)) };
    out
}
