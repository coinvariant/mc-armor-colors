use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::{env, thread};

fn red(rgb: u32) -> i32 {
    ((rgb >> 16) & 0xFF) as i32
}
fn green(rgb: u32) -> i32 {
    ((rgb >> 8) & 0xFF) as i32
}
fn blue(rgb: u32) -> i32 {
    (rgb & 0xFF) as i32
}

fn max3(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).max(c)
}

fn make_rgb(r: i32, g: i32, b: i32) -> u32 {
    let r = r.clamp(0, 255) as u32;
    let g = g.clamp(0, 255) as u32;
    let b = b.clamp(0, 255) as u32;
    (r << 16) | (g << 8) | b
}

fn color_id_from_rgb(rgb: u32) -> u32 {
    rgb
}

fn rgb_from_id(id: u32) -> (i32, i32, i32) {
    (red(id), green(id), blue(id))
}

const DYE_COLORS: [u32; 16] = [
    11546150, // Red
    16351261, // Orange
    16701501, // Yellow
    8439583,  // Lime
    6192150,  // Green
    3847130,  // Light Blue
    1481884,  // Cyan
    3949738,  // Blue
    8991416,  // Purple
    13061821, // Magenta
    15961002, // Pink
    16383998, // White
    10329495, // Light Gray
    4673362,  // Gray
    1908001,  // Black
    8606770,  // Brown
];

const MAX_DYES_PER_STEP: usize = 8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Pattern {
    sr: i32, // sum of R
    sg: i32, // sum of G
    sb: i32, // sum of B
    sm: i32, // sum of max(R,G,B) per dye
    k: u8,   // number of dyes in this pattern (1..=8)
}

// Generate all unique patterns of 1–8 dyes (multisets over 16 colors)
fn generate_patterns() -> Vec<Pattern> {
    let mut set: HashSet<Pattern> = HashSet::new();

    fn rec(
        idx: usize,
        picked: usize,
        sr: i32,
        sg: i32,
        sb: i32,
        sm: i32,
        set: &mut HashSet<Pattern>,
    ) {
        if picked > MAX_DYES_PER_STEP {
            return;
        }
        if idx == DYE_COLORS.len() {
            if picked >= 1 {
                set.insert(Pattern {
                    sr,
                    sg,
                    sb,
                    sm,
                    k: picked as u8,
                });
            }
            return;
        }

        let remaining_slots = MAX_DYES_PER_STEP - picked;

        let dye_rgb = DYE_COLORS[idx];
        let dr = red(dye_rgb);
        let dg = green(dye_rgb);
        let db = blue(dye_rgb);
        let dm = max3(dr, dg, db);

        for c in 0..=remaining_slots {
            let c_i32 = c as i32;
            let nsr = sr + c_i32 * dr;
            let nsg = sg + c_i32 * dg;
            let nsb = sb + c_i32 * db;
            let nsm = sm + c_i32 * dm;
            rec(idx + 1, picked + c, nsr, nsg, nsb, nsm, set);
        }
    }

    rec(0, 0, 0, 0, 0, 0, &mut set);
    let mut v: Vec<Pattern> = set.into_iter().collect();
    v.sort_by_key(|p| p.k);
    v
}

fn apply_pattern(prev_color: Option<u32>, pat: &Pattern) -> u32 {
    let mut n4 = pat.sr; // sum R
    let mut n5 = pat.sg; // sum G
    let mut n6 = pat.sb; // sum B
    let mut n7 = pat.sm; // sum max per color
    let mut n8 = pat.k as i32; // count

    if let Some(rgb) = prev_color {
        let (r0, g0, b0) = rgb_from_id(rgb);
        let m0 = max3(r0, g0, b0);
        n4 += r0;
        n5 += g0;
        n6 += b0;
        n7 += m0;
        n8 += 1;
    }

    let mut r_avg = n4 / n8;
    let mut g_avg = n5 / n8;
    let mut b_avg = n6 / n8;

    let f = (n7 as f32) / (n8 as f32); // average of maxima
    let f2 = max3(r_avg, g_avg, b_avg) as f32;

    if f2 > 0.0 {
        r_avg = ((r_avg as f32) * f / f2) as i32;
        g_avg = ((g_avg as f32) * f / f2) as i32;
        b_avg = ((b_avg as f32) * f / f2) as i32;
    } else {
        r_avg = 0;
        g_avg = 0;
        b_avg = 0;
    }

    make_rgb(r_avg, g_avg, b_avg)
}

struct Bitset {
    data: Vec<AtomicU8>,
}

impl Bitset {
    fn new(bits: usize) -> Self {
        let bytes = (bits + 7) / 8;
        Bitset {
            data: (0..bytes).map(|_| AtomicU8::new(0)).collect(),
        }
    }

    /// Returns true if the bit was already set, false if we just set it.
    fn check_and_set(&self, idx: u32) -> bool {
        let i = idx as usize;
        let byte = i / 8;
        let bit = i % 8;
        let mask = 1u8 << bit;
        let prev = self.data[byte].fetch_or(mask, Ordering::Relaxed);
        (prev & mask) != 0
    }

    fn count_ones(&self) -> u64 {
        self.data
            .iter()
            .map(|b| b.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}

#[derive(Clone, Copy)]
struct Edge {
    start_color: Option<u32>,
    end_color: u32,
    pattern: Pattern,
}

fn main() {
    let start_time = Instant::now();

    let args: Vec<String> = env::args().collect();
    let requested_threads = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1)
    } else {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    };
    let num_threads = requested_threads.max(1);
    println!("Using {} threads", num_threads);

    println!("Generating dye patterns...");
    let patterns = generate_patterns();
    println!("Number of unique patterns: {}", patterns.len());

    let visited = Bitset::new(1 << 24);
    let mut frontier: VecDeque<u32> = VecDeque::new();
    let mut edges: Vec<Edge> = Vec::new();

    println!("Computing colors from undyed armor (first dyeing)...");
    for pat in &patterns {
        let color = apply_pattern(None, pat);
        let id = color_id_from_rgb(color);
        if !visited.check_and_set(id) {
            frontier.push_back(id);
            edges.push(Edge {
                start_color: None,
                end_color: color,
                pattern: *pat,
            });
        }
    }

    let mut total = visited.count_ones();
    println!(
        "Initial reachable colors after first dye: {} (time: {:.2?})",
        total,
        start_time.elapsed()
    );

    let mut step = 0usize;
    let substep = AtomicU64::new(0);

    while !frontier.is_empty() {
        step += 1;
        let level_size = frontier.len();
        let level_start_time = Instant::now();
        println!("Level size {:3}", level_size);

        // Move the frontier into a Vec for chunking
        let current_frontier: Vec<u32> = frontier.drain(..).collect();

        let mut next_frontier: Vec<u32> = Vec::new();
        let mut level_edges: Vec<Edge> = Vec::new();

        let threads_for_level = num_threads.min(level_size.max(1));
        let chunk_size = (level_size + threads_for_level - 1) / threads_for_level;

        thread::scope(|scope| {
            let mut handles = Vec::new();

            for t in 0..threads_for_level {
                let start = t * chunk_size;
                if start >= level_size {
                    break;
                }
                let end = ((t + 1) * chunk_size).min(level_size);

                let slice = &current_frontier[start..end];
                let patterns_ref = &patterns;
                let visited_ref = &visited;
                let substep_ref = &substep;

                handles.push(scope.spawn(move || {
                    let mut local_next = Vec::new();
                    let mut local_edges = Vec::new();

                    for &current_id in slice {
                        let s = substep_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if s % 1000 == 0 {
                            println!("Substep {}", s);
                        }

                        for pat in patterns_ref {
                            let new_color = apply_pattern(Some(current_id), pat);
                            let new_id = color_id_from_rgb(new_color);
                            if !visited_ref.check_and_set(new_id) {
                                local_next.push(new_id);
                                local_edges.push(Edge {
                                    start_color: Some(current_id),
                                    end_color: new_color,
                                    pattern: *pat,
                                });
                            }
                        }
                    }

                    (local_next, local_edges)
                }));
            }

            for handle in handles {
                let (local_next, local_edges_local) = handle.join().unwrap();
                next_frontier.extend(local_next);
                level_edges.extend(local_edges_local);
            }
        });

        let added = next_frontier.len();
        total = visited.count_ones();

        println!(
            "Step {:3}: frontier processed = {:8}, new colors = {:8}, total = {:9}, step time = {:.2?}, total time = {:.2?}",
            step,
            level_size,
            added,
            total,
            level_start_time.elapsed(),
            start_time.elapsed()
        );

        if added == 0 {
            println!("No new colors found; search saturated.");
            break;
        }

        // Here you can write `level_edges` to disk per level if desired.
        // For now we just accumulate them:
        edges.extend(level_edges);

        // Prepare frontier for next level
        frontier = next_frontier.into_iter().collect();
    }

    println!(
        "Finished. Total reachable colors: {} (elapsed: {:.2?})",
        visited.count_ones(),
        start_time.elapsed()
    );
}
