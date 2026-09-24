use std::time::{Duration, Instant};

use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, TimestampKind, devices};

const USAGE: &str = "usage: spike-transcribe-cpp <model.gguf> <clip.wav>... [--runs N] [--cpu] [--device N]
  --device N   index into the enumerated Vulkan devices (default: backend's automatic choice)";

fn read_wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16_000, "{path}: need 16 kHz");
    assert_eq!(spec.channels, 1, "{path}: need mono");
    assert_eq!(spec.bits_per_sample, 16, "{path}: need 16-bit");
    r.samples::<i16>()
        .map(|s| f32::from(s.unwrap()) / 32768.0)
        .collect()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

fn main() {
    let mut runs = 5usize;
    let mut cpu = false;
    let mut device_ix: Option<usize> = None;
    let mut pos = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--runs" => runs = it.next().expect("--runs N").parse().expect("--runs N"),
            "--cpu" => cpu = true,
            "--device" => device_ix = Some(it.next().expect("--device N").parse().expect("--device N")),
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            _ => pos.push(a),
        }
    }
    if pos.len() < 2 {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let model_path = &pos[0];
    let clips = &pos[1..];

    println!(
        "transcribe-cpp {} ({})",
        transcribe_cpp::version(),
        transcribe_cpp::version_commit()
    );
    let all = devices();
    for d in &all {
        println!("device: {} [{}] {} id={:?}", d.name, d.kind, d.description, d.device_id);
    }

    let backend = if cpu { Backend::Cpu } else { Backend::Vulkan };
    let device = device_ix.map(|i| {
        all.iter()
            .filter(|d| d.kind == "vulkan")
            .nth(i)
            .expect("no such Vulkan device")
            .clone()
    });

    let t = Instant::now();
    let model = Model::load_with(model_path, &ModelOptions { backend, device })
        .unwrap_or_else(|e| panic!("load {model_path}: {e}"));
    let load = t.elapsed();
    let bound = model.device().map(|d| d.description).unwrap_or_else(|e| format!("? ({e})"));
    println!(
        "model: {} {} on backend={} device={} load={:.0} ms",
        model.arch(),
        model.variant(),
        model.backend(),
        bound,
        ms(load)
    );

    let t = Instant::now();
    let mut session = model.session().expect("session");
    println!("session create: {:.0} ms", ms(t.elapsed()));

    let opts = RunOptions {
        timestamps: TimestampKind::None,
        language: Some("en".into()),
        ..Default::default()
    };

    // Warm-up on the first clip so the first timed run does not pay pipeline/shader compile.
    let warm_pcm = read_wav(&clips[0]);
    let t = Instant::now();
    let r = session.run(&warm_pcm, &opts).expect("warm-up run");
    println!("warm-up ({}): {:.0} ms -> {:?}", clips[0], ms(t.elapsed()), r.text.trim());

    for clip in clips {
        let pcm = read_wav(clip);
        let secs = pcm.len() as f64 / 16_000.0;
        let mut times = Vec::with_capacity(runs);
        let mut text = String::new();
        let mut last_timings = None;
        for _ in 0..runs {
            let t = Instant::now();
            let r = session.run(&pcm, &opts).unwrap_or_else(|e| panic!("run {clip}: {e}"));
            times.push(ms(t.elapsed()));
            text = r.text.trim().to_string();
            last_timings = Some(r.timings);
        }
        let med = median(&times);
        let vals: Vec<String> = times.iter().map(|v| format!("{v:.1}")).collect();
        println!(
            "\nclip {clip} ({secs:.2} s)\n  runs ms: [{}]\n  median {med:.1} ms  RTF {:.4}\n  stages (last run): {:?}\n  text: {text:?}",
            vals.join(", "),
            med / 1000.0 / secs,
            last_timings.unwrap()
        );

        // An unseen length: ggml re-plans graphs per shape, so this is what a real utterance costs.
        let cut = &pcm[..pcm.len() * 9 / 10];
        let t = Instant::now();
        session.run(cut, &opts).unwrap_or_else(|e| panic!("run 90% {clip}: {e}"));
        println!("  90%-length unseen run: {:.1} ms", ms(t.elapsed()));
    }
}
