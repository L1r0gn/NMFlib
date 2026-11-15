///|
/// SplitMix64 PRNG (Steele et al., 2014). Deterministic and seedable; no
/// system entropy is involved, so fixed seeds reproduce results exactly.
/// All arithmetic wraps at 64 bits by MoonBit UInt64 semantics.
priv struct SplitMix64 {
  mut state : UInt64
}

///|
fn SplitMix64::new(seed : UInt64) -> SplitMix64 {
  { state: seed, }
}

///|
fn SplitMix64::next_u64(self : SplitMix64) -> UInt64 {
  self.state = self.state + 0x9E3779B97F4A7C15UL
  let mut z = self.state
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL
  z ^ (z >> 31)
}

///|
/// Uniform double in (0, 1]: `(u53 + 1) / 2^53`. Strictly positive so random
/// initialization never produces exact zeros (a zero factor would be stuck
/// at zero under multiplicative updates).
fn SplitMix64::next_double(self : SplitMix64) -> Double {
  ((self.next_u64() >> 11) + 1UL).to_double() / 9007199254740992.0
}
