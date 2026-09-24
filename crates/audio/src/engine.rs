//! Capture worker: drains the ring, resamples, routes audio to the recording or the
//! pre-roll, and owns the warm window. Single-threaded and clock-injected so every state
//! transition is unit-testable; the thread around it only forwards commands.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rtrb::{Consumer, RingBuffer};
use wl_core::recorder::{Recorder, RecorderConfig, RecorderError, Recording};
use wl_core::stt::SAMPLE_RATE;

use crate::capture::{CallbackShared, OpenedStream, StreamOpener};
use crate::resample::StreamResampler;

pub trait Clock: Send + 'static {
    fn now(&self) -> Instant;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Read by the handle without a round trip to the worker: the overlay polls the level at
/// frame rate and must never wait behind a stream open.
#[derive(Debug, Default)]
pub struct Shared {
    level_bits: AtomicU32,
    warm: AtomicBool,
}

impl Shared {
    pub fn level(&self) -> f32 {
        f32::from_bits(self.level_bits.load(Ordering::Relaxed))
    }
    pub fn is_warm(&self) -> bool {
        self.warm.load(Ordering::Relaxed)
    }
    fn set_level(&self, v: f32) {
        self.level_bits.store(v.to_bits(), Ordering::Relaxed);
    }
}

/// 50 ms at 16 kHz.
const LEVEL_WINDOW: usize = SAMPLE_RATE as usize / 20;
/// Seconds of device-rate audio the ring holds. The worker drains every 10 ms, so this only
/// matters when it is starved, e.g. while a stop flushes; overruns are counted, not hidden.
const RING_SECONDS: usize = 2;

struct LevelMeter {
    buf: Vec<f32>,
    pos: usize,
}

impl LevelMeter {
    fn new() -> Self {
        Self {
            buf: vec![0.0; LEVEL_WINDOW],
            pos: 0,
        }
    }
    fn push(&mut self, samples: &[f32]) {
        for &s in samples.iter().rev().take(LEVEL_WINDOW).rev() {
            self.buf[self.pos] = s;
            self.pos = (self.pos + 1) % LEVEL_WINDOW;
        }
    }
    fn rms(&self) -> f32 {
        let sum: f32 = self.buf.iter().map(|s| s * s).sum();
        (sum / LEVEL_WINDOW as f32).sqrt().min(1.0)
    }
    fn clear(&mut self) {
        self.buf.fill(0.0);
    }
}

struct Live {
    _stream: OpenedStream,
    rate: u32,
    consumer: Consumer<f32>,
    cb: Arc<CallbackShared>,
    resampler: StreamResampler,
}

struct Take {
    pcm: Vec<f32>,
    overruns: usize,
    lost_at: Option<Instant>,
    capped: bool,
    /// Delivery-rate audit, see [`Take::shortfall`].
    first_data_at: Option<Instant>,
    last_data_at: Option<Instant>,
    frames_after_first: u64,
}

impl Take {
    fn new(pcm: Vec<f32>) -> Self {
        Self {
            pcm,
            overruns: 0,
            lost_at: None,
            capped: false,
            first_data_at: None,
            last_data_at: None,
            frames_after_first: 0,
        }
    }

    fn saw_data(&mut self, frames: usize, now: Instant) {
        if frames == 0 {
            return;
        }
        if self.first_data_at.is_none() {
            // The first batch covers time before it was drained, so it is the reference
            // point rather than counted: that keeps cold-start latency out of the audit.
            self.first_data_at = Some(now);
        } else {
            self.frames_after_first += frames as u64;
        }
        self.last_data_at = Some(now);
    }

    /// Device-rate frames the device should have delivered but did not. Some virtual
    /// devices (measured: RDP "Remote Audio", 2026-09-24) deliver about three quarters of
    /// their nominal rate, flagging only discontinuities and keeping their own timestamps
    /// consistent with the short count, so wall-clock time is the only witness.
    fn shortfall(&self, rate: u32) -> u64 {
        let (Some(a), Some(b)) = (self.first_data_at, self.last_data_at) else {
            return 0;
        };
        let expected = b.saturating_duration_since(a).as_secs_f64() * rate as f64;
        let missing = expected - self.frames_after_first as f64;
        // Drain timing jitters by a callback period or two; below this it is noise.
        let tolerance = rate as f64 * 0.05 + expected * 0.02;
        if missing > tolerance {
            missing as u64
        } else {
            0
        }
    }
}

enum Mode {
    Idle { since: Instant },
    Recording(Take),
}

pub struct Engine<O: StreamOpener> {
    cfg: RecorderConfig,
    opener: O,
    live: Option<Live>,
    mode: Mode,
    preroll: VecDeque<f32>,
    preroll_cap: usize,
    max_samples: usize,
    scratch_in: Vec<f32>,
    scratch_out: Vec<f32>,
    meter: LevelMeter,
    shared: Arc<Shared>,
}

fn samples_for(d: Duration) -> usize {
    (d.as_secs_f64() * SAMPLE_RATE as f64).round() as usize
}

impl<O: StreamOpener> Engine<O> {
    pub fn new(cfg: RecorderConfig, opener: O, now: Instant) -> Self {
        let preroll_cap = samples_for(cfg.pre_roll);
        let max_samples = samples_for(cfg.max_duration);
        Self {
            cfg,
            opener,
            live: None,
            mode: Mode::Idle { since: now },
            preroll: VecDeque::with_capacity(preroll_cap),
            preroll_cap,
            max_samples,
            scratch_in: Vec::new(),
            scratch_out: Vec::new(),
            meter: LevelMeter::new(),
            shared: Arc::new(Shared::default()),
        }
    }

    pub fn shared(&self) -> Arc<Shared> {
        self.shared.clone()
    }

    pub fn is_live(&self) -> bool {
        self.live.is_some()
    }

    fn open(&mut self) -> Result<(), RecorderError> {
        let t0 = Instant::now();
        let cb = Arc::new(CallbackShared::default());
        // Capacity is fixed before the rate is known; 192 kHz covers every shared-mode
        // format Windows offers.
        let (producer, consumer) = RingBuffer::new(192_000 * RING_SECONDS);
        let stream = self
            .opener
            .open(self.cfg.device.as_deref(), producer, cb.clone())?;
        let resampler = StreamResampler::new(stream.sample_rate).map_err(RecorderError::Stream)?;
        tracing::info!(
            device = %stream.device_name,
            rate = stream.sample_rate,
            channels = stream.channels,
            open_ms = t0.elapsed().as_secs_f64() * 1000.0,
            "capture stream opened (cold start)"
        );
        self.live = Some(Live {
            rate: stream.sample_rate,
            _stream: stream,
            consumer,
            cb,
            resampler,
        });
        self.preroll.clear();
        self.shared.warm.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn close(&mut self, why: &str) {
        if self.live.take().is_some() {
            tracing::info!(reason = why, "capture stream closed");
        }
        self.preroll.clear();
        self.meter.clear();
        self.shared.set_level(0.0);
        self.shared.warm.store(false, Ordering::Relaxed);
    }

    /// Move everything captured so far to its destination and apply timeouts.
    pub fn pump(&mut self, now: Instant) {
        let lost = self
            .live
            .as_ref()
            .is_some_and(|l| l.cb.lost.load(Ordering::Acquire));
        self.drain(now);
        if lost {
            tracing::warn!("capture device lost or default device changed; stream closed");
            if let Mode::Recording(take) = &mut self.mode
                && take.lost_at.is_none()
            {
                take.lost_at = Some(now);
            }
            self.close("device lost");
            return;
        }
        if let Mode::Idle { since } = self.mode
            && self.live.is_some()
            && now.saturating_duration_since(since) >= self.cfg.warm_window
        {
            self.close("warm window elapsed");
        }
    }

    fn drain(&mut self, now: Instant) {
        let Some(live) = self.live.as_mut() else {
            return;
        };
        self.scratch_in.clear();
        self.scratch_out.clear();
        let n = live.consumer.slots();
        if let Ok(chunk) = live.consumer.read_chunk(n) {
            let (a, b) = chunk.as_slices();
            self.scratch_in.extend_from_slice(a);
            self.scratch_in.extend_from_slice(b);
            chunk.commit_all();
        }
        let dropped = live.cb.dropped.swap(0, Ordering::Relaxed);
        let rate = live.rate;
        live.resampler.push(&self.scratch_in, &mut self.scratch_out);
        if let Mode::Recording(take) = &mut self.mode {
            take.overruns += (dropped as u64 * SAMPLE_RATE as u64 / rate as u64) as usize;
            take.saw_data(self.scratch_in.len(), now);
        }
        self.route_out();
    }

    fn route_out(&mut self) {
        let out = std::mem::take(&mut self.scratch_out);
        if !out.is_empty() {
            self.meter.push(&out);
            self.shared.set_level(self.meter.rms());
        }
        match &mut self.mode {
            Mode::Recording(take) => {
                let room = self.max_samples.saturating_sub(take.pcm.len());
                if out.len() > room {
                    take.capped = true;
                }
                take.pcm.extend_from_slice(&out[..out.len().min(room)]);
            }
            Mode::Idle { .. } => {
                let keep = out.len().min(self.preroll_cap);
                self.preroll.extend(&out[out.len() - keep..]);
                let excess = self.preroll.len().saturating_sub(self.preroll_cap);
                self.preroll.drain(..excess);
            }
        }
        self.scratch_out = out;
        self.scratch_out.clear();
    }

    pub fn start(&mut self, now: Instant) -> Result<(), RecorderError> {
        if matches!(self.mode, Mode::Recording(_)) {
            return Err(RecorderError::State("already recording"));
        }
        if self.live.is_some() {
            self.pump(now);
        }
        if self.live.is_none() {
            self.open()?;
        }
        let mut pcm =
            Vec::with_capacity(self.max_samples.min(samples_for(Duration::from_secs(15))));
        pcm.extend(self.preroll.drain(..));
        pcm.truncate(self.max_samples);
        self.mode = Mode::Recording(Take::new(pcm));
        Ok(())
    }

    pub fn stop(&mut self, now: Instant) -> Result<Recording, RecorderError> {
        if !matches!(self.mode, Mode::Recording(_)) {
            return Err(RecorderError::State("not recording"));
        }
        self.pump(now);
        let mut short = 0;
        if let Some(live) = self.live.as_mut() {
            live.resampler.flush(&mut self.scratch_out);
            if let Mode::Recording(take) = &self.mode {
                let rate = live.rate;
                let missing = take.shortfall(rate);
                if missing > 0 {
                    short = (missing * SAMPLE_RATE as u64 / rate as u64) as usize;
                    tracing::warn!(
                        missing_ms = missing * 1000 / rate as u64,
                        "input device delivered less audio than its nominal rate"
                    );
                }
            }
            self.route_out();
        }
        let Mode::Recording(take) = std::mem::replace(&mut self.mode, Mode::Idle { since: now })
        else {
            unreachable!("checked above");
        };
        let lost_frames = take
            .lost_at
            .map(|t| samples_for(now.saturating_duration_since(t)))
            .unwrap_or(0);
        let rec = Recording {
            duration: Duration::from_secs_f64(take.pcm.len() as f64 / SAMPLE_RATE as f64),
            pcm: take.pcm,
            sample_rate: SAMPLE_RATE,
            dropped_frames: take.overruns + lost_frames + short,
            max_duration_reached: take.capped,
            device_lost: take.lost_at.is_some(),
        };
        // Zero warm window means "never keep the mic open", not "close on the next tick".
        self.pump(now);
        Ok(rec)
    }

    pub fn cancel(&mut self, now: Instant) {
        if matches!(self.mode, Mode::Recording(_)) {
            self.pump(now);
            self.mode = Mode::Idle { since: now };
            self.pump(now);
        }
    }
}

enum Cmd {
    Start(mpsc::Sender<Result<(), RecorderError>>),
    Stop(mpsc::Sender<Result<Recording, RecorderError>>),
    Cancel,
}

/// A [`Recorder`] that runs an [`Engine`] on its own thread. The stream is owned by that
/// thread for its whole life, so the COM apartment cpal sets up there is never shared with
/// the UI or hook threads.
pub struct WorkerRecorder {
    tx: Option<mpsc::Sender<Cmd>>,
    shared: Arc<Shared>,
    join: Option<JoinHandle<()>>,
}

const POLL: Duration = Duration::from_millis(10);

impl WorkerRecorder {
    pub fn spawn<O: StreamOpener, C: Clock>(
        cfg: RecorderConfig,
        opener: O,
        clock: C,
    ) -> Result<Self, RecorderError> {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let mut engine = Engine::new(cfg, opener, clock.now());
        let shared = engine.shared();
        let join = std::thread::Builder::new()
            .name("wl-audio".into())
            .spawn(move || {
                loop {
                    let cmd = if engine.is_live() {
                        match rx.recv_timeout(POLL) {
                            Ok(c) => Some(c),
                            Err(mpsc::RecvTimeoutError::Timeout) => None,
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    } else {
                        // Cold: nothing to drain, so sleep until asked instead of polling.
                        match rx.recv() {
                            Ok(c) => Some(c),
                            Err(_) => break,
                        }
                    };
                    let now = clock.now();
                    match cmd {
                        Some(Cmd::Start(reply)) => {
                            let _ = reply.send(engine.start(now));
                        }
                        Some(Cmd::Stop(reply)) => {
                            let _ = reply.send(engine.stop(now));
                        }
                        Some(Cmd::Cancel) => engine.cancel(now),
                        None => engine.pump(now),
                    }
                }
            })
            .map_err(|e| RecorderError::Stream(format!("cannot spawn audio thread: {e}")))?;
        Ok(Self {
            tx: Some(tx),
            shared,
            join: Some(join),
        })
    }

    fn send(&self, cmd: Cmd) -> Result<(), RecorderError> {
        self.tx
            .as_ref()
            .and_then(|tx| tx.send(cmd).ok())
            .ok_or(RecorderError::Stream("audio thread is gone".into()))
    }
}

fn gone<T>(_: T) -> RecorderError {
    RecorderError::Stream("audio thread is gone".into())
}

impl Recorder for WorkerRecorder {
    fn start(&mut self) -> Result<(), RecorderError> {
        let (tx, rx) = mpsc::channel();
        self.send(Cmd::Start(tx))?;
        rx.recv().map_err(gone)?
    }

    fn stop(&mut self) -> Result<Recording, RecorderError> {
        let (tx, rx) = mpsc::channel();
        self.send(Cmd::Stop(tx))?;
        rx.recv().map_err(gone)?
    }

    fn cancel(&mut self) {
        let _ = self.send(Cmd::Cancel);
    }

    fn level(&self) -> f32 {
        self.shared.level()
    }

    fn is_warm(&self) -> bool {
        self.shared.is_warm()
    }
}

impl Drop for WorkerRecorder {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::Producer;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Probe {
        producer: Arc<Mutex<Option<Producer<f32>>>>,
        cb: Arc<Mutex<Option<Arc<CallbackShared>>>>,
        opens: Arc<AtomicU32>,
        open_handles: Arc<AtomicU32>,
    }

    struct Handle(Arc<AtomicU32>);
    impl Drop for Handle {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct FakeOpener {
        rate: u32,
        probe: Probe,
        fail: bool,
    }

    impl StreamOpener for FakeOpener {
        fn open(
            &mut self,
            _device: Option<&str>,
            producer: Producer<f32>,
            shared: Arc<CallbackShared>,
        ) -> Result<OpenedStream, RecorderError> {
            if self.fail {
                return Err(RecorderError::NoInputDevice);
            }
            self.probe.opens.fetch_add(1, Ordering::SeqCst);
            self.probe.open_handles.fetch_add(1, Ordering::SeqCst);
            *self.probe.producer.lock().unwrap() = Some(producer);
            *self.probe.cb.lock().unwrap() = Some(shared);
            Ok(OpenedStream {
                sample_rate: self.rate,
                channels: 1,
                device_name: "fake".into(),
                handle: Box::new(Handle(self.probe.open_handles.clone())),
            })
        }
    }

    impl Probe {
        fn feed(&self, v: f32, n: usize) {
            let mut g = self.producer.lock().unwrap();
            let p = g.as_mut().expect("stream open");
            for _ in 0..n {
                p.push(v).expect("ring has room");
            }
        }
        fn cb(&self) -> Arc<CallbackShared> {
            self.cb.lock().unwrap().clone().unwrap()
        }
    }

    fn engine(cfg: RecorderConfig, rate: u32) -> (Engine<FakeOpener>, Probe, Instant) {
        let probe = Probe::default();
        let t0 = Instant::now();
        let e = Engine::new(
            cfg,
            FakeOpener {
                rate,
                probe: probe.clone(),
                fail: false,
            },
            t0,
        );
        (e, probe, t0)
    }

    /// Feed `secs` of constant `v` at 48 kHz in 10 ms blocks, pumping like the worker.
    fn feed_secs(e: &mut Engine<FakeOpener>, p: &Probe, v: f32, secs: f64, t: &mut Instant) {
        let blocks = (secs * 100.0).round() as usize;
        for _ in 0..blocks {
            p.feed(v, 480);
            *t += Duration::from_millis(10);
            e.pump(*t);
        }
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn cold_start_opens_warm_start_reuses() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        assert!(!e.shared().is_warm());
        e.start(t).unwrap();
        assert!(e.shared().is_warm());
        feed_secs(&mut e, &p, 0.1, 0.5, &mut t);
        let r = e.stop(t).unwrap();
        assert_eq!(
            r.pcm.len(),
            8000,
            "0.5 s at 16 kHz, no pre-roll on a cold start"
        );
        assert_eq!(r.sample_rate, 16_000);
        assert_eq!(r.duration, ms(500));
        assert!(e.shared().is_warm());
        e.start(t).unwrap();
        assert_eq!(p.opens.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn preroll_seeds_the_next_recording() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.0, 0.2, &mut t);
        e.stop(t).unwrap();
        // Idle but warm: the speaker starts just before the key press.
        feed_secs(&mut e, &p, 0.0, 1.0, &mut t);
        feed_secs(&mut e, &p, 0.5, 0.2, &mut t);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.5, 0.5, &mut t);
        let r = e.stop(t).unwrap();
        // The pre-roll ends where the resampler's output did at `start()`, which trails the
        // ring by at most one 20 ms chunk plus the FFT delay; that audio is not lost, it
        // lands in the recording instead.
        let extra = r.pcm.len() - (4800 + 8000);
        assert!(extra <= 400, "{}", r.pcm.len());
        // Pre-roll holds the last 300 ms: 100 ms of silence then 200 ms of signal.
        let head: f32 = r.pcm[..1000].iter().map(|x| x.abs()).sum::<f32>() / 1000.0;
        let tail: f32 = r.pcm[2400..4800].iter().sum::<f32>() / 2400.0;
        assert!(head < 0.05, "head {head}");
        assert!((tail - 0.5).abs() < 0.02, "tail {tail}");
    }

    #[test]
    fn warm_window_closes_stream_then_reopens() {
        let cfg = RecorderConfig {
            warm_window: Duration::from_secs(30),
            ..Default::default()
        };
        let (mut e, p, mut t) = engine(cfg, 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.1, 0.1, &mut t);
        e.stop(t).unwrap();
        let stopped = t;
        e.pump(stopped + Duration::from_secs(29));
        assert!(e.shared().is_warm());
        assert_eq!(p.open_handles.load(Ordering::SeqCst), 1);
        e.pump(stopped + Duration::from_secs(30));
        assert!(!e.shared().is_warm());
        assert_eq!(p.open_handles.load(Ordering::SeqCst), 0, "stream dropped");
        assert_eq!(e.shared().level(), 0.0);
        e.start(stopped + Duration::from_secs(31)).unwrap();
        assert_eq!(p.opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn recording_is_never_closed_by_the_warm_window() {
        let cfg = RecorderConfig {
            warm_window: ms(100),
            ..Default::default()
        };
        let (mut e, p, mut t) = engine(cfg, 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.1, 1.0, &mut t);
        assert!(e.shared().is_warm());
        e.stop(t).unwrap();
        e.pump(t + ms(100));
        assert!(!e.shared().is_warm());
    }

    #[test]
    fn zero_warm_window_closes_on_stop_and_cancel() {
        let cfg = RecorderConfig {
            warm_window: Duration::ZERO,
            ..Default::default()
        };
        let (mut e, _p, t) = engine(cfg, 48_000);
        e.start(t).unwrap();
        e.stop(t).unwrap();
        assert!(!e.shared().is_warm());
        e.start(t).unwrap();
        e.cancel(t);
        assert!(!e.shared().is_warm());
    }

    #[test]
    fn cancel_discards_and_stays_warm() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.3, 0.5, &mut t);
        e.cancel(t);
        assert!(e.shared().is_warm());
        assert!(matches!(e.stop(t), Err(RecorderError::State(_))));
    }

    #[test]
    fn max_duration_truncates_and_flags() {
        let cfg = RecorderConfig {
            max_duration: Duration::from_secs(1),
            ..Default::default()
        };
        let (mut e, p, mut t) = engine(cfg, 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.1, 1.5, &mut t);
        let r = e.stop(t).unwrap();
        assert_eq!(r.pcm.len(), 16_000);
        assert!(r.max_duration_reached);
    }

    #[test]
    fn device_loss_keeps_audio_marks_drops_and_reopens() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.2, 0.5, &mut t);
        p.cb().lost.store(true, Ordering::Release);
        e.pump(t);
        assert!(!e.shared().is_warm());
        t += ms(500);
        let r = e.stop(t).unwrap();
        assert!(r.device_lost);
        // Without a flush the resampler's in-flight tail is lost with the stream.
        assert!(
            r.pcm.len() >= 7500 && r.pcm.len() <= 8000,
            "{}",
            r.pcm.len()
        );
        assert_eq!(r.dropped_frames, 8000);
        e.start(t).unwrap();
        assert_eq!(p.opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn ring_overruns_count_as_dropped_frames() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        e.start(t).unwrap();
        p.cb().dropped.store(4800, Ordering::Relaxed);
        feed_secs(&mut e, &p, 0.1, 0.1, &mut t);
        let r = e.stop(t).unwrap();
        assert_eq!(r.dropped_frames, 1600);
        assert!(!r.device_lost);
    }

    #[test]
    fn slow_device_is_reported_as_dropped() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        e.start(t).unwrap();
        // 75 % of the nominal rate, as the RDP redirected microphone delivers.
        for _ in 0..200 {
            p.feed(0.1, 360);
            t += ms(10);
            e.pump(t);
        }
        let r = e.stop(t).unwrap();
        let expected_missing = 2 * 16_000 / 4;
        assert!(
            r.dropped_frames.abs_diff(expected_missing) <= 200,
            "{}",
            r.dropped_frames
        );
    }

    #[test]
    fn on_time_device_reports_nothing_dropped() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 44_100);
        e.start(t).unwrap();
        // Bursty but complete delivery: 20 ms every other tick.
        for i in 0..300 {
            if i % 2 == 0 {
                p.feed(0.1, 882);
            }
            t += ms(10);
            e.pump(t);
        }
        assert_eq!(e.stop(t).unwrap().dropped_frames, 0);
    }

    #[test]
    fn state_errors() {
        let (mut e, _p, t) = engine(RecorderConfig::default(), 48_000);
        assert!(matches!(e.stop(t), Err(RecorderError::State(_))));
        e.start(t).unwrap();
        assert!(matches!(e.start(t), Err(RecorderError::State(_))));
    }

    #[test]
    fn open_failure_surfaces() {
        let probe = Probe::default();
        let t = Instant::now();
        let mut e = Engine::new(
            RecorderConfig::default(),
            FakeOpener {
                rate: 48_000,
                probe,
                fail: true,
            },
            t,
        );
        assert_eq!(e.start(t), Err(RecorderError::NoInputDevice));
        assert!(!e.shared().is_warm());
    }

    #[test]
    fn level_reflects_last_50ms() {
        let (mut e, p, mut t) = engine(RecorderConfig::default(), 48_000);
        e.start(t).unwrap();
        feed_secs(&mut e, &p, 0.5, 0.2, &mut t);
        assert!((e.shared().level() - 0.5).abs() < 0.02);
        feed_secs(&mut e, &p, 0.0, 0.1, &mut t);
        assert!(e.shared().level() < 0.01);
    }

    #[test]
    fn threaded_recorder_round_trip() {
        let probe = Probe::default();
        let mut r = WorkerRecorder::spawn(
            RecorderConfig::default(),
            FakeOpener {
                rate: 44_100,
                probe: probe.clone(),
                fail: false,
            },
            SystemClock,
        )
        .unwrap();
        r.start().unwrap();
        assert!(r.is_warm());
        probe.feed(0.25, 44_100 / 2);
        let rec = r.stop().unwrap();
        assert_eq!(rec.pcm.len(), 8000);
        drop(r);
        assert_eq!(probe.open_handles.load(Ordering::SeqCst), 0);
    }
}
