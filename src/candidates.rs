//! Index-based candidate streaming for the GPU search.
//!
//! Mirrors the CPU generator in `main.rs` (`permutations` of the known words +
//! `insert_missing` to fill 12-word phrases), but yields fixed `[u16; 12]`
//! arrays of BIP-39 word indices — the compact form the GPU kernel consumes.

use itertools::Itertools;
use std::collections::HashSet;

/// Streams every candidate as 12 word indices: permutations of the `known`
/// indices with `12 - known.len()` slots filled from `0..wordlist_len`.
///
/// `pinned` holds indices *into `known`* whose word stays fixed at that
/// position instead of being permuted — the GPU-side counterpart of a
/// `"quoted"` word in word mode (see `split_pinned_words` in `main.rs`).
pub fn stream(known: Vec<u16>, pinned: Vec<usize>, wordlist_len: usize) -> impl Iterator<Item = [u16; 12]> + Send {
    let n = known.len();
    let missing = 12 - n;

    let pinned_set: HashSet<usize> = pinned.iter().copied().collect();
    let pinned_vals: Vec<(usize, u16)> = pinned.iter().map(|&i| (i, known[i])).collect();
    let movable: Vec<u16> = known.iter().enumerate()
        .filter(|(i, _)| !pinned_set.contains(i))
        .map(|(_, &w)| w)
        .collect();
    let movable_len = movable.len();

    movable
        .into_iter()
        .permutations(movable_len)
        .flat_map(move |perm| {
            let mut base = vec![0u16; n];
            for &(i, w) in &pinned_vals { base[i] = w; }
            let mut mi = 0;
            for (i, slot) in base.iter_mut().enumerate() {
                if pinned_set.contains(&i) { continue; }
                *slot = perm[mi];
                mi += 1;
            }
            insert_missing(base, missing, wordlist_len)
        })
        .map(|v| {
            let mut a = [0u16; 12];
            a.copy_from_slice(&v);
            a
        })
}

/// Lazily inserts `remaining` indices from `0..wordlist_len` into every gap of
/// `seq`. `remaining` is at most 2, so the recursion is shallow.
fn insert_missing(
    seq: Vec<u16>,
    remaining: usize,
    wordlist_len: usize,
) -> Box<dyn Iterator<Item = Vec<u16>> + Send> {
    if remaining == 0 {
        return Box::new(std::iter::once(seq));
    }
    let len = seq.len();
    Box::new((0..=len).flat_map(move |pos| {
        let seq = seq.clone();
        (0..wordlist_len as u16).flat_map(move |word| {
            let mut next = Vec::with_capacity(seq.len() + 1);
            next.extend_from_slice(&seq[..pos]);
            next.push(word);
            next.extend_from_slice(&seq[pos..]);
            insert_missing(next, remaining - 1, wordlist_len)
        })
    }))
}
