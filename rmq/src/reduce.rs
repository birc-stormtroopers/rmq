use super::{RMQ, Point};
use super::tables::{powers,powers::Pow};
use super::sparse::Sparse;

/// Reduce an array x to the smallest value in each block (of size block_size)
/// and the index in the original array that this minimal value sits at.
pub fn reduce_array(x: &[usize], block_size: usize) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = Vec::new();
    let mut values: Vec<usize> = Vec::new();
    let no_blocks = x.len() / block_size;
    for block in 0..no_blocks {
        let begin = block * block_size;
        let end = begin + block_size;
        let Point(pos, val) = 
            Point::new(super::rmq(&x, begin, end).unwrap(), &x);
        indices.push(pos);
        values.push(val);
    }
    (indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce() {
        let bs = 3;
        let v = vec![3, 2, 6, 1, 7, 3, 10, 1, 6, 2, 1, 7, 0, 2];
        let (idx, val) = reduce_array(&v, bs);
        for (i, &pos) in idx.iter().enumerate() {
            assert_eq!(v[pos], val[i]);
        }
    }
}

pub struct Reduced<'a> {
    x: &'a [usize],
    reduced_vals: Vec<usize>,
    reduced_idx: Vec<usize>,
    tbl: Sparse<'a>
}

impl<'a> Reduced<'a> {
    #[allow(dead_code)] // only used in tests right now
    pub fn new(x: &'a [usize]) -> Self {
        let n = x.len();
        let Pow(block_size) = powers::log2_up(n);
        let (reduced_vals, reduced_idx) = reduce_array(x, block_size);
        let tbl = Sparse::new(&reduced_vals);
        Reduced{ x, reduced_vals, reduced_idx, tbl }
    }
}

impl<'a> RMQ for Reduced<'a> {
        fn rmq(&self, i: usize, j: usize) -> Option<usize> {
        // FIXME The naive solution we already have in super:: as a function
        super::rmq(self.x, i, j)
    }
}