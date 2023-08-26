use jpeg_decoder::Decoder;
use std::env;
use std::ops::Add;
use std::{error::Error, fs::File, io::BufReader};

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let transpose = args.contains(&"--transpose".to_string());

    let img_file = File::open(args.last().unwrap())?;
    let mut decoder = Decoder::new(BufReader::new(img_file));

    let pixels;
    let metadata;
    let width;
    let height;

    if transpose {
        let raw_pixels = decoder.decode()?;
        metadata = decoder.info().unwrap();
        let raw_height = metadata.height as usize;
        let raw_width = metadata.width as usize;

        let mut scratch = vec![0u8; raw_pixels.len()];
        for x in 0..raw_width {
            for y in 0..raw_height {
                let n_src = x + y * raw_width;
                let n_dst = y + x * raw_height;
                scratch[n_dst] = raw_pixels[n_src];
            }
        }

        width = metadata.height as usize;
        height = metadata.width as usize;
        pixels = scratch;
    }else{
        pixels = decoder.decode()?;
        metadata = decoder.info().unwrap();
        width = metadata.width as usize;
        height = metadata.height as usize;
    }


    let mut fft_planner = RealFftPlanner::<f64>::new();
    let fft_plan = fft_planner.plan_fft_forward(width);
    let mut fft_scratch = fft_plan.make_scratch_vec();

    let fft_scale = 1. / (width as f64).sqrt();

    let rows: Vec<Vec<Complex<f64>>> = (0..height)
        .map(|y| (y as usize * width * 3) as usize)
        .map(|n| &pixels[n..(n + (width * 3))])
        .map(|s| {
            let mut out_vec = Vec::with_capacity(width);
            for i in (0..s.len()).step_by(3) {
                out_vec.push((s[i] as f64 + s[i + 1] as f64 + s[i + 2] as f64) / 3.0);
            }
            out_vec
        })
        .map(|mut v| {
            let mut out = fft_plan.make_output_vec();
            //TODO: the input being mutable seems like a bug
            fft_plan
                .process_with_scratch(&mut v, &mut out, &mut fft_scratch)
                .expect("fft error");

            for e in &mut out {
                *e = e.scale(fft_scale);
            }

            out
        })
        .collect();

    let mut sums: Vec<Complex<f64>> = vec![Complex::default(); width];
    let final_scale = 1. / height as f64;

    for r in rows {
        for (v, f) in r.iter().zip(sums.iter_mut()) {
            *f = f.add(v);
        }
    }

    let magnitudes: Vec<f64> = sums.iter().map(|s| s.norm() * final_scale).collect();

    for (n, mag) in (0..).zip(magnitudes) {
        if mag > 0.0 {
            let wavelength = width as f64 / (n + 1) as f64;
            let sqrt_w = (wavelength as f64).sqrt();
            // we give the magnitude as the raw magnitude over the square root of wavelength
            // to normalize the magnitude across wavelengths since real world images
            // seem to follow a similar distribution of energies as noise
            println!("{}, {}", wavelength, mag / sqrt_w);
        }
    }

    Ok(())
}
