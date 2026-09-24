use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hush_audio::vad::{EnergyVad, Vad, has_speech, trim_silence};
use hush_audio::{CpalRecorder, Recorder, RecorderConfig, list_input_devices};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hush_audio=debug".into()),
        )
        .init();

    let mut out = None;
    let mut seconds = 5.0f64;
    let mut device = None;
    let mut list_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => seconds = args.next().ok_or("--seconds needs a value")?.parse()?,
            "--device" => device = Some(args.next().ok_or("--device needs a value")?),
            "--list" => list_only = true,
            _ => out = Some(a),
        }
    }

    let devices = list_input_devices()?;
    println!("input devices: {}", devices.len());
    for d in &devices {
        println!(
            "  {} {:<40} {:<24} {}",
            if d.is_default { "*" } else { " " },
            d.name,
            d.default_format.as_deref().unwrap_or("?"),
            d.id
        );
    }
    if list_only {
        return Ok(());
    }
    let out = out.ok_or("usage: record <out.wav> [--seconds N] [--device NAME] [--list]")?;

    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;

    let mut rec = CpalRecorder::new(RecorderConfig {
        device,
        ..Default::default()
    })?;

    let t0 = Instant::now();
    rec.start()?;
    println!(
        "start() took {:.1} ms (cold)",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    let deadline = t0 + Duration::from_secs_f64(seconds);
    while Instant::now() < deadline && !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));
        let l = rec.level();
        let db = 20.0 * l.max(1e-6).log10();
        let bar = ((db + 60.0).max(0.0) / 60.0 * 40.0) as usize;
        println!("{:>6.1} dBFS |{:<40}|", db, "#".repeat(bar));
    }
    let t1 = Instant::now();
    let r = rec.stop()?;
    println!(
        "stop() took {:.1} ms; {:.2} s captured, dropped {} frames, device lost: {}, capped: {}",
        t1.elapsed().as_secs_f64() * 1000.0,
        r.duration.as_secs_f64(),
        r.dropped_frames,
        r.device_lost,
        r.max_duration_reached
    );

    let t2 = Instant::now();
    rec.start()?;
    println!(
        "warm start() took {:.2} ms",
        t2.elapsed().as_secs_f64() * 1000.0
    );
    rec.cancel();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: r.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&out, spec)?;
    for &s in &r.pcm {
        w.write_sample(s)?;
    }
    w.finalize()?;
    println!("wrote {out}");

    let mut vad = EnergyVad::default();
    let mut events = vad.push(&r.pcm);
    events.extend(vad.finish());
    for e in &events {
        println!("  {e:?}");
    }
    let trimmed = trim_silence(&r.pcm, &events, Duration::from_millis(200));
    println!(
        "speech: {}; trimmed to {:.2} s",
        has_speech(&events),
        trimmed.len() as f64 / r.sample_rate as f64
    );
    Ok(())
}
