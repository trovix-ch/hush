mod wav;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use hush_core::stt::{Backend, DecodeOptions, SAMPLE_RATE, SttEngine};
use hush_stt::{TranscribeCppEngine, TranscribeCppOptions, models, transcribe_cpp};

const USAGE: &str = "\
usage: bench-stt <wav-path> [--engine transcribe-cpp|ort] [--backend vulkan|cpu|directml]
                 [--device N] [--runs N] [--language en] [--model <id>]
                 [--model-dir <path>] [--threads N] [--joint-cpu]
       bench-stt --list-devices

  --engine     transcribe-cpp (default) or ort (only in builds with the `ort-engine`
               feature)
  --backend    default vulkan for transcribe-cpp, directml for ort
  --device     transcribe-cpp: index among Vulkan devices (see --list-devices);
               ort: DXGI adapter index. Default: the runtime's choice
  --model      manifest id; default parakeet-tdt-0.6b-v3-f16-gguf for transcribe-cpp;
               for ort parakeet-tdt-0.6b-v3 on directml, -int8 on cpu
  --model-dir  load from this directory instead of the default models root
  --threads    CPU threads (default: half the logical cores)
  --joint-cpu  ort only: run the decoder/joint graph on the CPU even on directml

After the timed runs, one more run on the first 90% of the clip reports what an
utterance of a length the engine has not seen before costs.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    TranscribeCpp,
    Ort,
}

struct Args {
    wav: PathBuf,
    engine: Engine,
    backend: Option<Backend>,
    runs: usize,
    language: Option<String>,
    model: Option<String>,
    model_dir: Option<PathBuf>,
    joint_on_cpu: bool,
    threads: Option<usize>,
    device: Option<usize>,
}

fn parse_args() -> Result<Option<Args>> {
    let mut it = std::env::args().skip(1);
    let mut wav = None;
    let mut args = Args {
        wav: PathBuf::new(),
        engine: Engine::TranscribeCpp,
        backend: None,
        runs: 5,
        language: None,
        model: None,
        model_dir: None,
        joint_on_cpu: false,
        threads: None,
        device: None,
    };
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match a.as_str() {
            "--engine" => {
                args.engine = match value("--engine")?.to_ascii_lowercase().as_str() {
                    "transcribe-cpp" | "ggml" => Engine::TranscribeCpp,
                    "ort" => Engine::Ort,
                    other => bail!("unknown engine `{other}`\n{USAGE}"),
                }
            }
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
            "--device" => args.device = Some(value("--device")?.parse().context("--device")?),
            "--joint-cpu" => args.joint_on_cpu = true,
            "--list-devices" => {
                for d in transcribe_cpp::vulkan_devices() {
                    println!(
                        "vulkan {}: {} ({}) id={} {:.1} GiB",
                        d.index,
                        d.description,
                        d.name,
                        d.device_id.as_deref().unwrap_or("?"),
                        d.memory_total as f64 / (1u64 << 30) as f64
                    );
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

/// The lines describe the runtime that actually loaded.
fn build_engine(
    args: &Args,
    backend: Backend,
    load_path: &std::path::Path,
) -> Result<(Box<dyn SttEngine>, Vec<String>)> {
    match args.engine {
        Engine::TranscribeCpp => {
            let engine = TranscribeCppEngine::with_options(
                load_path,
                TranscribeCppOptions {
                    backend,
                    gpu_device: args.device,
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
            gpu_device: args
                .device
                .map(|d| i32::try_from(d).context("--device"))
                .transpose()?,
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
        Engine::TranscribeCpp => Backend::Vulkan,
        Engine::Ort => Backend::DirectMl,
    });
    let model_id = args.model.clone().unwrap_or_else(|| {
        match (args.engine, backend) {
            (Engine::TranscribeCpp, _) => models::DEFAULT_MODEL_ID,
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
    let mut last = None;
    for _ in 0..args.runs {
        let tr = engine.transcribe(&audio, &opts)?;
        times.push(tr.inference_time);
        last = Some(tr);
    }
    let last = last.expect("runs >= 1");
    let med = median(&times);

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
