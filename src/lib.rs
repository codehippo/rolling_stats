//! # RollingStats
//!
//! `rolling_stats` is a (primarily) no_std Rust library for maintaining a rolling window of
//! statistical data and performing various statistical calculations on that data.
//!
//! ## Features
//!
//! - Maintain a fixed-size rolling window of `i32` values
//! - Calculate mean and standard deviation of the window
//! - Sample new values based on current statistics
//! - Handle byte input with configurable endianness
//!
//! ## Feature Flags
//! The `rolling_stats` crate has the following cargo feature flags:
//! - `std`
//!     + Optional, disabled by default
//!     + Implements the `std::io::Write` trait
//!
//! ## Usage
//!
//! You can use the library in your Rust code like this:
//!
//! ```rust
//! use rolling_stats::RollingStats;
//! use endi::Endian;
//!
//! let mut stats = RollingStats::<3>::try_new(Endian::Little, 42).unwrap();
//! stats.push_bytes_to_window(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
//!
//! println!("Mean: {}", stats.mean());
//! println!("Standard Deviation: {}", stats.std_dev());
//! println!("Sampled Value: {}", stats.sample_value());
//! ```
//!
//! For more detailed information, see the documentation for the `RollingStats` struct.

#![cfg_attr(not(test), no_std)]
#[cfg(feature = "std")]
extern crate std;

use arraydeque::{ArrayDeque, Wrapping};
use arrayvec::ArrayVec;
use core::mem;
use const_format::formatcp;
use endi::Endian;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

const I32_SIZE_BYTES: usize = mem::size_of::<i32>();
const APPROX_DEFAULT_STACK_LIMIT_IN_BYTES: usize = 128 * 1_024; // 128 KiB
const MAX_DEFAULT_WINDOW_SIZE: usize = APPROX_DEFAULT_STACK_LIMIT_IN_BYTES / I32_SIZE_BYTES;

/// A struct that maintains a rolling window of i32 values and provides statistical calculations.
///
/// The struct uses a fixed-size window to store the most recent values. It can handle
/// byte input of various endianness and provides methods to calculate mean, standard deviation,
/// and sample new values based on the current statistics.
///
/// # Type Parameters
///
/// * `WINDOW_SIZE`: The maximum number of i32 values to keep in the rolling window.
pub struct RollingStats<const WINDOW_SIZE: usize> {
    endianness: Endian,
    window: ArrayDeque<i32, WINDOW_SIZE, Wrapping>,
    buffer: ArrayVec<u8, I32_SIZE_BYTES>,
    rng: ChaCha8Rng,
}

impl<const WINDOW_SIZE: usize> RollingStats<WINDOW_SIZE> {
    /// Attempts to create a new `RollingStats` instance with the specified endianness and random number generator seed.
    ///
    /// # Arguments
    ///
    /// * `endianness`: The byte order to use when reading i32 values from byte slices.
    /// * `seed`: A seed value for the internal random number generator.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` if the `WINDOW_SIZE` is within the default library-defined stack limit.
    /// * `Err(&'static str)` if the `WINDOW_SIZE` exceeds the default stack limit, containing an error message.
    ///
    /// # Examples
    ///
    /// ```
    /// use rolling_stats::RollingStats;
    /// use endi::Endian;
    ///
    /// let stats = RollingStats::<32>::try_new(Endian::Little, 42);
    /// assert!(stats.is_ok());
    /// ```
    pub fn try_new(endianness: Endian, seed: u64) -> Result<Self, &'static str> {
        if WINDOW_SIZE > MAX_DEFAULT_WINDOW_SIZE {
            Err(formatcp!("You exceeded the default window size. It would exceed the default library-defined stack limit of {} KiB.", APPROX_DEFAULT_STACK_LIMIT_IN_BYTES / 1024))
        } else {
            Ok(Self {
                endianness,
                window: ArrayDeque::new(),
                buffer: ArrayVec::new(),
                rng: ChaCha8Rng::seed_from_u64(seed),
            })
        }
    }

    /// Attempts to create a new `RollingStats` instance with the specified endianness, random number generator seed,
    /// and a custom stack limit.
    ///
    /// # Arguments
    ///
    /// * `endianness`: The byte order to use when reading i32 values from byte slices.
    /// * `seed`: A seed value for the internal random number generator.
    /// * `stack_limit_in_bytes`: The custom stack limit in bytes to use for determining the maximum window size.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` if the `WINDOW_SIZE` is within the specified custom stack limit.
    /// * `Err(&'static str)` if the `WINDOW_SIZE` exceeds the custom stack limit, containing an error message.
    ///
    /// # Examples
    ///
    /// ```
    /// use rolling_stats::RollingStats;
    /// use endi::Endian;
    ///
    /// const CUSTOM_STACK_LIMIT: usize = 1024 * 1024; // 1 MiB
    /// let stats = RollingStats::<1000>::try_new_with_stack_limit::<CUSTOM_STACK_LIMIT>(Endian::Little, 42);
    /// assert!(stats.is_ok());
    /// ```
    pub fn try_new_with_stack_limit<const CUSTOM_STACK_LIMIT: usize>(endianness: Endian, seed: u64) -> Result<Self, &'static str> {
        let max_custom_window_size = CUSTOM_STACK_LIMIT / I32_SIZE_BYTES;

        if WINDOW_SIZE > max_custom_window_size {
            Err("You exceeded the maximum window size with respect to your custom stack limit.")
        } else {
            Ok(Self {
                endianness,
                window: ArrayDeque::new(),
                buffer: ArrayVec::new(),
                rng: ChaCha8Rng::seed_from_u64(seed),
            })
        }
    }

    /// Calculates the mean of the values in the current window.
    ///
    /// # Returns
    ///
    /// The mean as an f32, or NaN if the window is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rolling_stats::RollingStats;
    /// use endi::Endian;
    ///
    /// let mut stats = RollingStats::<3>::try_new(Endian::Little, 42).unwrap();
    /// stats.push_bytes_to_window(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    ///
    /// assert_eq!(stats.mean(), 2.0);
    /// ```
    pub fn mean(&self) -> f32 {
        let sum: i32 = self.window.iter().sum();
        let num_count: usize = self.window.len();

        if num_count == 0 {
            return f32::NAN;
        }

        sum as f32 / num_count as f32
    }

    /// Calculates the standard deviation of the values in the current window.
    ///
    /// # Returns
    ///
    /// The standard deviation as an f32, or NaN if the window is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rolling_stats::RollingStats;
    /// use endi::Endian;
    ///
    /// let mut stats = RollingStats::<3>::try_new(Endian::Little, 42).unwrap();
    /// stats.push_bytes_to_window(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    ///
    /// assert!((stats.std_dev() - 0.81649658092773).abs() < 1e-6);
    /// ```
    pub fn std_dev(&self) -> f32 {
        let mean = self.mean();
        let num_count = self.window.len();

        if num_count == 0 {
            return f32::NAN;
        }

        let variance: f32 = self
            .window
            .iter()
            .map(|&x| (x as f32 - mean).powi(2))
            .sum::<f32>()
            / num_count as f32;

        variance.sqrt()
    }

    /// Samples a new value based on the current mean and standard deviation.
    ///
    /// This method uses a normal distribution with the current mean and standard deviation
    /// to generate a new sample value.
    ///
    /// # Returns
    ///
    /// A new sample value as an f32, or NaN if sampling is not possible (e.g., empty window).
    ///
    /// # Examples
    ///
    /// ```
    /// use rolling_stats::RollingStats;
    /// use endi::Endian;
    ///
    /// let mut stats = RollingStats::<3>::try_new(Endian::Little, 42).unwrap();
    /// stats.push_bytes_to_window(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    ///
    /// let sample = stats.sample_value();
    /// assert!(sample.is_finite());
    /// ```
    pub fn sample_value(&mut self) -> f32 {
        match Normal::new(self.mean(), self.std_dev()) {
            Ok(distribution) => distribution.sample(&mut self.rng),
            Err(_) => f32::NAN,
        }
    }

    fn try_fill_buffer<'a>(&mut self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        let remaining = self.buffer.remaining_capacity();
        let (to_buffer, remaining_bytes) = bytes.split_at(remaining.min(bytes.len()));

        // This line can't panic when unwrapped because we're using the minimum of remaining
        // capacity and bytes slice length.
        self.buffer.try_extend_from_slice(to_buffer).unwrap();

        if self.buffer.is_full() {
            self.window
                .push_front(self.endianness.read_i32(self.buffer.as_slice()));
            self.buffer.clear();

            (!remaining_bytes.is_empty()).then_some(remaining_bytes)
        } else {
            None
        }
    }

    /// Pushes a slice of bytes to the window, interpreting them as i32 values.
    ///
    /// This method reads i32 values from the input byte slice using the specified endianness,
    /// and adds them to the rolling window. If the input doesn't align perfectly with i32 boundaries,
    /// the remaining bytes are buffered for the next call.
    ///
    /// # Arguments
    ///
    /// * `bytes`: A slice of bytes to be interpreted as i32 values and added to the window.
    ///
    /// # Examples
    ///
    /// ```
    /// use rolling_stats::RollingStats;
    /// use endi::Endian;
    ///
    /// let mut stats = RollingStats::<3>::try_new(Endian::Little, 42).unwrap();
    /// stats.push_bytes_to_window(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    ///
    /// assert_eq!(stats.mean(), 2.0);
    /// ```
    pub fn push_bytes_to_window(&mut self, bytes: &[u8]) {
        let remaining_bytes = match self.try_fill_buffer(bytes) {
            None => return,
            Some(remaining_bytes) => remaining_bytes,
        };

        remaining_bytes.chunks(I32_SIZE_BYTES).for_each(|chunk| {
            if chunk.len() == I32_SIZE_BYTES {
                let number = self.endianness.read_i32(chunk);
                self.window.push_front(number);
            } else {
                self.try_fill_buffer(chunk);
            }
        });
    }
}

#[cfg(feature = "std")]
impl<const WINDOW_SIZE: usize> std::io::Write for RollingStats<WINDOW_SIZE> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.push_bytes_to_window(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rv::misc::ks_test;
    use rv::prelude::*;

    const SEED: u64 = u64::MAX / 4;

    #[test]
    fn test_initialization() {
        let stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        assert_eq!(stats.window.len(), 0);
        assert_eq!(stats.buffer.len(), 0);
        assert!(matches!(stats.endianness, Endian::Big));
    }

    #[test]
    fn test_empty_state() {
        let stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        assert!(stats.mean().is_nan());
        assert!(stats.std_dev().is_nan());
    }

    #[test]
    fn test_push_complete_numbers() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert_eq!(stats.window.len(), 3);
        assert_eq!(
            stats.window.iter().cloned().collect::<ArrayVec<i32, 3>>(),
            ArrayVec::<i32, 3>::from([3, 2, 1])
        );
    }

    #[test]
    fn test_mean_calculation() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert_eq!(stats.mean(), 2.0);
    }

    #[test]
    fn test_mean_calculation_with_less_numbers() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        let mut stats_long = RollingStats::<10>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        stats_long.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert_eq!(stats.mean(), stats_long.mean());
    }

    #[test]
    fn test_std_dev_calculation() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert!((stats.std_dev() - 0.81649658092773).abs() < 1e-6);
    }

    #[test]
    fn test_std_dev_calculation_with_less_numbers() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        let mut stats_long = RollingStats::<10>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        stats_long.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert!((stats.std_dev() - stats_long.std_dev()).abs() < 1e-6);
    }

    #[test]
    fn test_window_rolling() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4]);
        assert_eq!(
            stats.window.iter().cloned().collect::<ArrayVec<i32, 3>>(),
            ArrayVec::<i32, 3>::from([4, 3, 2])
        );
        assert_eq!(stats.mean(), 3.0);
    }

    #[test]
    fn test_incomplete_number_handling() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0]);
        assert_eq!(stats.window.len(), 1);
        assert_eq!(stats.buffer.len(), 2);
        stats.push_bytes_to_window(&[0, 2, 0, 0, 0, 3]);
        assert_eq!(stats.window.len(), 3);
        assert_eq!(
            stats.window.iter().cloned().collect::<ArrayVec<i32, 3>>(),
            ArrayVec::<i32, 3>::from([3, 2, 1])
        );
    }

    #[test]
    fn test_little_endian_handling() {
        let mut stats = RollingStats::<3>::try_new(Endian::Little, SEED).unwrap();
        stats.push_bytes_to_window(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(
            stats.window.iter().cloned().collect::<ArrayVec<i32, 3>>(),
            ArrayVec::<i32, 3>::from([3, 2, 1])
        );
    }

    #[test]
    fn test_huge_window_size() {
        let stats = RollingStats::<{ u16::MAX as usize }>::try_new(Endian::Big, SEED);
        assert!(stats.is_err());
    }

    #[test]
    fn test_initialization_with_increased_stack_limit() {
        const CUSTOM_STACK_LIMIT: usize = 512 * 1024; // 512 KiB
        let stats = RollingStats::<{ u16::MAX as usize }>::try_new_with_stack_limit::<CUSTOM_STACK_LIMIT>(Endian::Big, SEED);
        assert!(stats.is_ok());
    }

    #[test]
    fn test_sample_values() {
        let mut stats = RollingStats::<3>::try_new(Endian::Big, SEED).unwrap();
        stats.push_bytes_to_window(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);

        let mut sample_values = ArrayVec::<f64, 32>::new();
        for _ in 0..32 {
            sample_values.push(stats.sample_value() as f64)
        }

        let normal_distribution =
            Gaussian::new(stats.mean() as f64, stats.std_dev() as f64).unwrap();
        let normal_cdf = |x: f64| normal_distribution.cdf(&x);

        // Perform Kolmogorov-Smirnov test
        // This test compares the cumulative distribution function (CDF) of the sample
        // against the CDF of a specified theoretical distribution (in this case, normal).
        // It quantifies the maximum distance between these two CDFs.
        let (_, p_normal) = ks_test(sample_values.as_slice(), normal_cdf);

        // A p-value > 0.05 suggests the sample is likely drawn from the specified normal distribution
        assert!(p_normal > 0.05);
    }
}
