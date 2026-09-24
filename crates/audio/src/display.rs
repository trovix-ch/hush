//! Speech RMS sits around -30 to -15 dBFS, which is 0.03 to 0.18 on a linear scale, so a
//! bar driven by the linear value barely moves. The display uses a dB scale instead; the
//! linear level stays as it is for anything that thresholds on energy.

use std::time::Duration;

const FLOOR_DBFS: f32 = -50.0;
const CEILING_DBFS: f32 = -10.0;
const ATTACK: Duration = Duration::from_millis(30);
const RELEASE: Duration = Duration::from_millis(300);
/// Below this the bar is drawn empty, so a decaying release stops changing the picture.
const SNAP_TO_ZERO: f32 = 0.01;

/// `rms` in 0..=1 linear, mapped to 0..=1 on a dB scale from -50 dBFS to -10 dBFS.
pub fn display_level(rms: f32) -> f32 {
    if rms.is_nan() || rms <= 0.0 {
        return 0.0;
    }
    let dbfs = 20.0 * rms.log10();
    ((dbfs - FLOOR_DBFS) / (CEILING_DBFS - FLOOR_DBFS)).clamp(0.0, 1.0)
}

/// Fast attack, slow release, so a syllable shows at once and the bar does not flicker
/// between them.
#[derive(Debug, Default, Clone, Copy)]
pub struct LevelBallistics {
    value: f32,
}

impl LevelBallistics {
    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    /// `rms` is the linear level; `dt` the time since the previous update.
    pub fn update(&mut self, rms: f32, dt: Duration) -> f32 {
        let target = display_level(rms);
        let tau = if target > self.value { ATTACK } else { RELEASE };
        let k = 1.0 - (-dt.as_secs_f32() / tau.as_secs_f32()).exp();
        self.value += (target - self.value) * k;
        if self.value < SNAP_TO_ZERO {
            self.value = 0.0;
        }
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_at(dbfs: f32) -> f32 {
        10f32.powf(dbfs / 20.0)
    }

    #[test]
    fn maps_dbfs_to_the_display_scale() {
        for (db, want) in [(-50.0, 0.0), (-30.0, 0.5), (-20.0, 0.75), (-10.0, 1.0)] {
            let got = display_level(rms_at(db));
            assert!((got - want).abs() < 1e-3, "{db} dBFS -> {got}, want {want}");
        }
    }

    #[test]
    fn clamps_outside_the_range_and_handles_silence() {
        assert_eq!(display_level(0.0), 0.0);
        assert_eq!(display_level(f32::NAN), 0.0);
        assert_eq!(display_level(rms_at(-70.0)), 0.0);
        assert_eq!(display_level(1.0), 1.0);
    }

    #[test]
    fn attacks_fast_and_releases_slowly() {
        let tick = Duration::from_millis(50);
        let mut b = LevelBallistics::default();
        let up = b.update(rms_at(-20.0), tick);
        assert!(up > 0.55, "attack after one tick: {up}");
        let down = b.update(0.0, tick);
        assert!(down > up * 0.8, "release after one tick: {down}");
        for _ in 0..40 {
            b.update(0.0, tick);
        }
        assert_eq!(b.update(0.0, tick), 0.0);
    }
}
