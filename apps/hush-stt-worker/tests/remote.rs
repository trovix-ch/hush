use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hush_core::CancelToken;
use hush_core::stt::{Backend, DecodeOptions, SAMPLE_RATE, SttEngine, SttError};
use hush_stt::protocol::parse_backend;
use hush_stt::{RemoteEngine, RemoteOptions};

fn worker() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_hush-stt-worker"))
}

/// The CPU unless `HUSH_TEST_TC_BACKEND` says otherwise, so the test runs without a GPU.
fn live_engine() -> Option<RemoteEngine> {
    let model = std::env::var_os("HUSH_TEST_TC_MODEL")?;
    let backend = std::env::var("HUSH_TEST_TC_BACKEND")
        .ok()
        .and_then(|b| parse_backend(&b))
        .unwrap_or(Backend::Cpu);
    Some(RemoteEngine::new(RemoteOptions::new(worker(), model.into(), backend)).unwrap())
}

fn silence(secs: usize) -> Vec<f32> {
    vec![0.0; SAMPLE_RATE as usize * secs]
}

#[test]
fn a_bad_model_path_is_a_load_error_not_a_hang() {
    let started = Instant::now();
    let err = RemoteEngine::new(RemoteOptions::new(
        worker(),
        Path::new("does-not-exist.gguf").into(),
        Backend::Cpu,
    ))
    .err()
    .unwrap();
    assert!(matches!(err, SttError::Load(_)), "{err}");
    assert!(err.to_string().contains("does-not-exist"), "{err}");
    assert!(started.elapsed() < Duration::from_secs(30));
}

#[test]
fn a_crash_mid_inference_is_backend_died_and_the_next_call_succeeds() {
    let Some(mut engine) = live_engine() else {
        return;
    };
    let first_pid = engine.worker_pid().unwrap();
    assert!(
        engine
            .transcribe(&silence(1), &DecodeOptions::default())
            .is_ok()
    );

    engine.arm_crash(Duration::from_millis(30)).unwrap();
    let started = Instant::now();
    let r = engine.transcribe(&silence(30), &DecodeOptions::default());
    eprintln!("crashed call returned after {:?}: {r:?}", started.elapsed());
    assert!(matches!(r, Err(SttError::BackendDied(_))), "{r:?}");

    let started = Instant::now();
    engine.wait_ready(None).unwrap();
    eprintln!("replacement ready after {:?}", started.elapsed());
    let second_pid = engine.worker_pid().unwrap();
    assert_ne!(first_pid, second_pid);
    assert_eq!(engine.restarts(), 1);
    let tr = engine.transcribe(&silence(2), &DecodeOptions::default());
    assert!(tr.is_ok(), "{tr:?}");
    assert_eq!(engine.worker_info().unwrap().id, engine.info().id);
}

#[test]
fn a_death_while_idle_is_repaired_by_the_next_call() {
    let Some(mut engine) = live_engine() else {
        return;
    };
    let pid = engine.worker_pid().unwrap();
    let killed = std::process::Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output()
        .unwrap();
    assert!(killed.status.success(), "{killed:?}");
    let r = engine.transcribe(&silence(1), &DecodeOptions::default());
    assert!(matches!(r, Err(SttError::BackendDied(_))), "{r:?}");
    let r = engine.transcribe(&silence(1), &DecodeOptions::default());
    assert!(r.is_ok(), "{r:?}");
    assert_ne!(engine.worker_pid(), Some(pid));
}

#[test]
fn cancel_crosses_the_pipe_and_leaves_the_worker_usable() {
    let Some(mut engine) = live_engine() else {
        return;
    };
    let cancel = CancelToken::new();
    let opts = DecodeOptions {
        cancel: cancel.clone(),
        ..Default::default()
    };
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        cancel.cancel();
    });
    let r = engine.transcribe(&silence(30), &opts);
    canceller.join().unwrap();
    assert!(matches!(r, Err(SttError::Cancelled)), "{r:?}");

    let expired = DecodeOptions {
        cancel: CancelToken::new().with_deadline(Instant::now()),
        ..Default::default()
    };
    assert!(matches!(
        engine.transcribe(&silence(1), &expired),
        Err(SttError::Deadline)
    ));
    // Finished work may still come back after a deadline, as in process; a deadline must
    // never read as a user's cancel.
    let late = DecodeOptions {
        cancel: CancelToken::new().with_timeout(Duration::from_millis(20)),
        ..Default::default()
    };
    let r = engine.transcribe(&silence(30), &late);
    assert!(matches!(r, Ok(_) | Err(SttError::Deadline)), "{r:?}");
    let pid = engine.worker_pid();
    assert!(
        engine
            .transcribe(&silence(1), &DecodeOptions::default())
            .is_ok()
    );
    assert_eq!(engine.worker_pid(), pid, "a cancel must not cost a restart");
    assert_eq!(engine.restarts(), 0);
}
