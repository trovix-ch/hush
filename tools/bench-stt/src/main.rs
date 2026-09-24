mod wav;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use hush_core::gpu::{self, GpuRequest, GpuSelector};
use hush_core::stt::{Backend, DecodeOptions, SAMPLE_RATE, SttEngine};
use hush_stt::{RemoteEngine, RemoteOptions, models};
use hush_stt_worker::{TranscribeCppEngine, TranscribeCppOptions, transcribe_cpp};

const USAGE: &str = "\
usage: bench-stt <wav-path> [--engine transcribe-cpp|remote|ort] [--backend vulkan|cpu|directml]
                 [--gpu auto|<pci>|<name>] [--runs N] [--language en] [--model <id>]
                 [--model-dir <path>] [--threads N] [--joint-cpu] [--worker <path>]
                 [--gap-ms N] [--kick] [--kick-lead-ms N] [--kick-every-ms N]
       bench-stt --list-devices

  --engine     transcribe-cpp (default, in this process), remote (the same engine in
               hush-stt-worker, as the app runs it) or ort (only in builds with the
               `ort-engine` feature)
  --worker     remote only: the worker executable (default: next to bench-stt)
  --backend    default vulkan for transcribe-cpp and remote, directml for ort
  --gpu        transcribe-cpp and remote on Vulkan: auto (default; the discrete card
               with the most free memory), a PCI bus id such as 05:00, or part of the
               device name, as hush's engine.device takes. ort uses the default adapter
  --model      manifest id; default parakeet-tdt-0.6b-v3-f16-gguf for transcribe-cpp;
               for ort parakeet-tdt-0.6b-v3 on directml, -int8 on cpu
  --model-dir  load from this directory instead of the default models root
  --threads    CPU threads (default: half the logical cores)
  --joint-cpu  ort only: run the decoder/joint graph on the CPU even on directml
  --gap-ms     idle this long before every timed run, as dictation does between
               utterances (default 0: back to back)
  --kick       call the engine's nudge before every timed run, as the app does at
               key-down, then wait --kick-lead-ms (default 1000) before transcribing
  --kick-every-ms  with --kick, nudge again at this interval during the lead

After the timed runs, one more run on the first 90% of the clip reports what an
utterance of a length the engine has not seen before costs.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    TranscribeCpp,
    Remote,
    Ort,
}

struct Args {
    wav: PathBuf,
    engine: Engine,
    worker: Option<PathBuf>,
    backend: Option<Backend>,
    runs: usize,
    language: Option<String>,
    model: Option<String>,
    model_dir: Option<PathBuf>,
    joint_on_cpu: bool,
    threads: Option<usize>,
    gpu: GpuSelector,
    gap: Duration,
    kick: bool,
    kick_lead: Duration,
    kick_every: Option<Duration>,
}

fn millis(v: String, name: &str) -> Result<Duration> {
    Ok(Duration::from_millis(v.parse().context(name.to_string())?))
}

fn parse_args() -> Result<Option<Args>> {
    let mut it = std::env::args().skip(1);
    let mut wav = None;
    let mut args = Args {
        wav: PathBuf::new(),
        engine: Engine::TranscribeCpp,
        worker: None,
        backend: None,
        runs: 5,
        language: None,
        model: None,
        model_dir: None,
        joint_on_cpu: false,
        threads: None,
        gpu: GpuSelector::Auto,
        gap: Duration::ZERO,
        kick: false,
        kick_lead: Duration::from_secs(1),
        kick_every: None,
    };
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match a.as_str() {
            "--engine" => {
                args.engine = match value("--engine")?.to_ascii_lowercase().as_str() {
                    "transcribe-cpp" | "ggml" => Engine::TranscribeCpp,
                    "remote" => Engine::Remote,
                    "ort" => Engine::Ort,
                    other => bail!("unknown engine `{other}`\n{USAGE}"),
                }
            }
            "--worker" => args.worker = Some(value("--worker")?.into()),
            "--backend" => {
                args.backend = Some(match value("--backend")?.to_ascii_lowercase().as_str() {
                    "vulkan" => Backend::Vulkan,
                    "directml" | "dml" => Backend::DirectMl,
                    "cpu" => Backend::Cpu,
                    other => bail!("unknown backend `{other}`\n{USAGE}"),
                })
            }
            "--runs" => args.runs = value("--runs")?.parse().context("--runs")?,
            "--language" => args.language = Some(value("--language")?),
            "--model" => args.model = Some(value("--model")?),
            "--model-dir" => args.model_dir = Some(value("--model-dir")?.into()),
            "--threads" => args.threads = Some(value("--threads")?.parse().context("--threads")?),
            "--gpu" => {
                args.gpu = value("--gpu")?
                    .parse()
                    .map_err(|e: String| anyhow::anyhow!("--gpu: {e}"))?;
            }
            "--joint-cpu" => args.joint_on_cpu = true,
            "--gap-ms" => args.gap = millis(value("--gap-ms")?, "--gap-ms")?,
            "--kick" => args.kick = true,
            "--kick-lead-ms" => {
                args.kick_lead = millis(value("--kick-lead-ms")?, "--kick-lead-ms")?;
            }
            "--kick-every-ms" => {
                let every = millis(value("--kick-every-ms")?, "--kick-every-ms")?;
                if every.is_zero() {
                    bail!("--kick-every-ms must be positive");
                }
                args.kick_every = Some(every);
            }
            "--list-devices" => {
                for d in transcribe_cpp::vulkan_devices() {
                    println!("{}", d.table_row());
                }
                return Ok(None);
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(None);
            }
            s if s.starts_with('-') => bail!("unknown option `{s}`\n{USAGE}"),
            s if wav.is_none() => wav = Some(PathBuf::from(s)),
            s => bail!("unexpected argument `{s}`\n{USAGE}"),
        }
    }
    args.wav = wav.with_context(|| USAGE.to_string())?;
    if args.runs == 0 {
        bail!("--runs must be at least 1");
    }
    Ok(Some(args))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn median(v: &[Duration]) -> Duration {
    let mut s = v.to_vec();
    s.sort();
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2
    }
}

/// The PCI id `--gpu` names, by the same enumeration and policy hush uses.
fn resolve_gpu(selector: &GpuSelector) -> Result<String> {
    let devices = transcribe_cpp::vulkan_devices();
    let r = gpu::resolve(&GpuRequest::Selector(selector.clone()), &devices)
        .map_err(|e| anyhow::anyhow!("--gpu: {e}"))?;
    let d = &devices[r.order[0]];
    eprintln!("gpu: {} ({})", d.label(), r.why);
    d.pci
        .clone()
        .with_context(|| format!("{} reports no PCI bus id", d.label()))
}

/// The lines describe the runtime that actually loaded.
fn build_engine(
    args: &Args,
    backend: Backend,
    load_path: &std::path::Path,
) -> Result<(Box<dyn SttEngine>, Vec<String>)> {
    let gpu = match (args.engine, backend) {
        (Engine::TranscribeCpp | Engine::Remote, Backend::Vulkan) => Some(resolve_gpu(&args.gpu)?),
        _ => None,
    };
    match args.engine {
        Engine::TranscribeCpp => {
            let engine = TranscribeCppEngine::with_options(
                load_path,
                TranscribeCppOptions {
                    backend,
                    gpu,
                    threads: args.threads,
                },
            )?;
            let threads = args.threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).max(1))
            });
            let notes = vec![
                format!("transcribe.cpp: {}", transcribe_cpp::library_version()),
                format!("cpu threads:    {threads}"),
            ];
            Ok((Box::new(engine), notes))
        }
        Engine::Remote => {
            let worker = match &args.worker {
                Some(w) => w.clone(),
                None => hush_stt::remote::default_worker_path()?,
            };
            let engine = RemoteEngine::new(RemoteOptions {
                gpu,
                threads: args.threads,
                ..RemoteOptions::new(worker.clone(), load_path.to_path_buf(), backend)
            })?;
            let notes = vec![
                format!(
                    "worker:         {} (pid {})",
                    worker.display(),
                    engine.worker_pid().unwrap_or_default()
                ),
                format!(
                    "worker load:    {:.1} ms model, {:.1} ms warm-up (inside the worker)",
                    ms(engine.load_time()),
                    ms(engine.warm_up_time())
                ),
            ];
            Ok((Box::new(engine), notes))
        }
        Engine::Ort => build_ort(args, backend, load_path),
    }
}

#[cfg(feature = "ort-engine")]
fn build_ort(
    args: &Args,
    backend: Backend,
    load_path: &std::path::Path,
) -> Result<(Box<dyn SttEngine>, Vec<String>)> {
    let engine = hush_stt::ParakeetEngine::with_options(
        load_path,
        hush_stt::ParakeetOptions {
            backend,
            joint_on_cpu: args.joint_on_cpu,
            intra_threads: args.threads,
            gpu_device: None,
        },
    )?;
    let notes = vec![
        format!("onnxruntime:    {}", hush_stt::onnxruntime_build_info()),
        format!("joint on cpu:   {}", args.joint_on_cpu),
    ];
    Ok((Box::new(engine), notes))
}

#[cfg(not(feature = "ort-engine"))]
fn build_ort(
    _args: &Args,
    _backend: Backend,
    _load_path: &std::path::Path,
) -> Result<(Box<dyn SttEngine>, Vec<String>)> {
    bail!("this build has no ONNX Runtime engine; rebuild with `--features ort-engine`")
}

/// Stands in for key-down: nudge, then the user speaks for `lead` before releasing.
fn kick(engine: &mut dyn SttEngine, lead: Duration, every: Option<Duration>) -> Result<()> {
    let start = Instant::now();
    let t = Instant::now();
    engine.nudge()?;
    eprintln!("nudge: {:.1} ms", ms(t.elapsed()));
    if let Some(every) = every {
        let mut next = every;
        while next < lead {
            std::thread::sleep(next.saturating_sub(start.elapsed()));
            engine.nudge()?;
            next += every;
        }
    }
    std::thread::sleep(lead.saturating_sub(start.elapsed()));
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,transcribe_cpp=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let Some(args) = parse_args()? else {
        return Ok(());
    };

    let pcm = wav::read_wav(&args.wav)?;
    let audio = wav::to_engine_pcm(&pcm, SAMPLE_RATE)?;
    let audio_secs = audio.len() as f64 / f64::from(SAMPLE_RATE);

    let backend = args.backend.unwrap_or(match args.engine {
        Engine::TranscribeCpp | Engine::Remote => Backend::Vulkan,
        Engine::Ort => Backend::DirectMl,
    });
    let model_id = args.model.clone().unwrap_or_else(|| {
        match (args.engine, backend) {
            (Engine::TranscribeCpp | Engine::Remote, _) => models::DEFAULT_MODEL_ID,
            (Engine::Ort, Backend::Cpu) => "parakeet-tdt-0.6b-v3-int8",
            (Engine::Ort, _) => "parakeet-tdt-0.6b-v3",
        }
        .to_string()
    });
    let manifest = models::find(&model_id)?;
    let dir = match &args.model_dir {
        Some(d) => d.clone(),
        None => {
            let d = models::default_model_dir(&manifest.id)?;
            models::ensure_downloaded(manifest, &d)?;
            d
        }
    };
    let load_path = manifest.load_path(&dir);

    let t = Instant::now();
    let (mut engine, notes) = build_engine(&args, backend, &load_path)?;
    let load = t.elapsed();

    let t = Instant::now();
    engine.warm_up()?;
    let warm = t.elapsed();

    let opts = DecodeOptions {
        language: args.language.clone(),
        ..Default::default()
    };
    let mut times = Vec::with_capacity(args.runs);
    let mut calls = Vec::with_capacity(args.runs);
    let mut last = None;
    for i in 0..args.runs {
        std::thread::sleep(args.gap);
        if args.kick {
            kick(engine.as_mut(), args.kick_lead, args.kick_every)?;
        }
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let t = Instant::now();
        let tr = engine.transcribe(&audio, &opts)?;
        let call = t.elapsed();
        eprintln!(
            "run {i}: start {unix_ms} unix ms, {:.1} ms, call {:.1} ms",
            ms(tr.inference_time),
            ms(call)
        );
        times.push(tr.inference_time);
        calls.push(call);
        last = Some(tr);
    }
    let last = last.expect("runs >= 1");
    let med = median(&times);
    // For the worker this is the pipe: audio out, transcript back, heartbeats aside.
    let overheads: Vec<Duration> = calls
        .iter()
        .zip(&times)
        .map(|(c, t)| c.saturating_sub(*t))
        .collect();

    // Dictation never repeats a length, so a backend that specialises per input shape
    // looks better on repeated runs than it will in use.
    let novel = &audio[..audio.len() * 9 / 10];
    let novel_time = engine
        .transcribe(novel, &opts)
        .ok()
        .map(|t| t.inference_time);

    let info = engine.info();
    println!("wav:            {}", args.wav.display());
    println!(
        "input:          {} Hz, {} ch -> {} Hz mono",
        pcm.sample_rate, pcm.channels, SAMPLE_RATE
    );
    println!("model:          {} ({})", info.id, load_path.display());
    println!(
        "backend:        {:?} (requested {:?})",
        info.backend, backend
    );
    println!(
        "device:         {}",
        info.device.as_deref().unwrap_or("(not reported)")
    );
    for n in notes {
        println!("{n}");
    }
    println!("model load:     {:.1} ms", ms(load));
    println!("warm-up:        {:.1} ms", ms(warm));
    println!(
        "runs (ms):      {}",
        times
            .iter()
            .map(|d| format!("{:.1}", ms(*d)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("median:         {:.1} ms", ms(med));
    println!(
        "call median:    {:.1} ms (overhead over inference: median {:.2} ms, max {:.2} ms)",
        ms(median(&calls)),
        ms(median(&overheads)),
        ms(overheads.iter().max().copied().unwrap_or_default())
    );
    println!(
        "gap:            {} ms{}",
        args.gap.as_millis(),
        if args.kick {
            format!(
                ", kick {} ms before each run{}",
                args.kick_lead.as_millis(),
                args.kick_every
                    .map(|e| format!(", again every {} ms", e.as_millis()))
                    .unwrap_or_default()
            )
        } else {
            String::new()
        }
    );
    match novel_time {
        Some(d) => println!("novel length:   {:.1} ms (first 90% of the clip)", ms(d)),
        None => println!("novel length:   n/a (clip too short)"),
    }
    println!("audio:          {audio_secs:.2} s");
    println!("rtf:            {:.4}", med.as_secs_f64() / audio_secs);
    println!("segments:       {}", last.segments.len());
    println!(
        "language:       {}",
        last.language.as_deref().unwrap_or("-")
    );
    println!("text:           {}", last.text);
    Ok(())
}
