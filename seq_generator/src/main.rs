use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use strsim::levenshtein;

const NUCLEOTIDES: &[char] = &['A', 'C', 'T'];
const LENGTH: usize = 20;
const SEED: &str = "TTACACTCCATCCACTCAA";

fn ok_gc_content(combination: &[usize]) -> bool {
    let mut c_count = 0;
    let mut a_count = 0;
    for c in combination {
        if *c == 0 {
            a_count += 1;
        } else if *c == 1 {
            c_count += 1;
        }
    }
    let a_percent = (a_count as f32) / (LENGTH as f32);
    let c_percent = (c_count as f32) / (LENGTH as f32);
    (0.45..=0.55).contains(&c_percent) && a_percent < 0.28
}

fn has_forbidden_substrings(s: &str) -> bool {
    s.contains("CCC") || s.contains("TTTT") || s.contains("AAAA")
}

// 4 nonconsecutive Cs in any 6 consecutive nucleotides in the first 12 positions of a probe should be avoided.
fn has_nonconsecutive_c(combination: &[usize]) -> bool {
    let mut curr_count = combination[0..6].iter().filter(|&&i| i == 1).count();
    for i in 0..7 {
        if curr_count >= 4 {
            return true;
        }

        if combination[i] == 1 {
            curr_count -= 1;
        }
        if combination[i + 6] == 1 {
            curr_count += 1;
        }
    }
    false
}
fn generate_combinations(output_file: &str, first_twos: (usize, usize)) -> std::io::Result<()> {
    let mut file = File::create(output_file)?;
    let mut combination: Vec<usize> = vec![0; LENGTH];
    let first_char = first_twos.0;
    let second_char = first_twos.1;
    combination[0] = first_char;
    combination[1] = second_char;

    let mut i = 0;
    loop {
        let current = combination
            .iter()
            .map(|&j| NUCLEOTIDES[j])
            .collect::<String>();

        if !has_forbidden_substrings(&current)
            && ok_gc_content(&combination)
            && !has_nonconsecutive_c(&combination)
            && levenshtein(&current, SEED) > 7
        {
            if i % 10000 == 1 {
                println!("{} combinations generated", i);
            }
            i += 1;
            file.write_all(
                format!(
                    ">{}{}_{}\n{}\n",
                    NUCLEOTIDES[first_char],
                    NUCLEOTIDES[second_char],
                    i,
                    current.clone()
                )
                .as_bytes(),
            )?;
        }

        let mut carry = true;
        for i in (2..LENGTH).rev() {
            if carry {
                combination[i] += 1;
                if combination[i] == NUCLEOTIDES.len() {
                    combination[i] = 0;
                } else {
                    carry = false;
                }
            } else {
                break;
            }
        }

        if carry {
            break;
        }
    }

    Ok(())
}

fn main() {
    let nucleotide_combinations: Vec<(usize, usize)> = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .collect::<Vec<_>>();

    nucleotide_combinations.par_iter().for_each(|&x| {
        let file_name = format!("combinations_{}{}.txt", x.0, x.1);
        let result = generate_combinations(&file_name, x);

        if let Err(e) = result {
            eprintln!("Error writing to file {}: {}", file_name, e);
        }
    });
}
