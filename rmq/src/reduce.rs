use super::{RMQ, Point};
use super::tables::{powers,powers::Pow};
use super::sparse::Sparse;
use std::cmp;

// Using special types for accessing blocks, so we don't mix up
// indices in the full space with indices in the smaller space.
#[derive(Clone, Copy, Debug)]
pub struct BlockSize(pub usize);
#[derive(Clone, Copy, Debug)]
pub struct BlockIdx(pub usize);

pub fn block_size(n: usize) -> BlockSize {
    // The block size is log2(n) rounded up.
    let Pow(block_size) = powers::log2_up(n);
    BlockSize(block_size)
}

impl std::cmp::PartialEq for BlockIdx {
    #[inline]
    fn eq(&self, other: &BlockIdx) -> bool {
        self.0 == other.0
    }
}

impl std::cmp::PartialOrd for BlockIdx {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let BlockIdx(i) = *self;
        let BlockIdx(j) = *other;
        Some(i.cmp(&j))
    }
}


/// For index i and block size bs, compute (r,r*n) where r
/// is i/bs rounded down. That is, r is i divided by bs
/// rounded down, and r*bs is i adjusted downwards to the
/// closest multiple of bs.
pub fn round_down(i: usize, bs: BlockSize) -> (BlockIdx, usize) {
    let BlockSize(bs) = bs;
    let r = i / bs;
    (BlockIdx(r), r * bs)
}

/// For i and block size bs, compute (r,r*i) where r
/// is i/bs rounded up. That is, r is i divided by bs
/// rounded down, and r*bs is i adjusted upwards to the
/// closest multiple of bs.
pub fn round_up(i: usize, bs: BlockSize) -> (BlockIdx, usize) {
    let BlockSize(bs) = bs;
    let r = (i + bs - 1) / bs;
    (BlockIdx(r), r * bs)
}

/// Reduce an array x to the smallest value in each block (of size block_size)
/// and the index in the original array that this minimal value sits at.
pub fn reduce_array(x: &[usize], block_size: BlockSize) -> (Vec<usize>, Vec<usize>) {
    let BlockSize(bs) = block_size;
    let mut indices: Vec<usize> = Vec::new();
    let mut values: Vec<usize> = Vec::new();
    let no_blocks = x.len() / bs;
    for block in 0..no_blocks {
        let begin = block * bs;
        let end = begin + bs;
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
        let bs = BlockSize(3);
        let v = vec![3, 2, 6, 1, 7, 3, 10, 1, 6, 2, 1, 7, 0, 2];
        let (idx, val) = reduce_array(&v, bs);
        for (i, &pos) in idx.iter().enumerate() {
            assert_eq!(v[pos], val[i]);
        }
    }
}


use ouroboros::self_referencing;
#[self_referencing]
struct _Reduced<'a> {
    // Original data
    x: &'a [usize],

    // For the sparse stuff...
    block_size: BlockSize,
    reduced_vals: Vec<usize>,
    reduced_idx: Vec<usize>,
    #[borrows(reduced_vals)]
    #[covariant]
    sparse: Sparse<'this>,
}
pub struct Reduced<'a>(_Reduced<'a>);


impl<'a> Reduced<'a> {
    #[allow(dead_code)] // only used in tests right now
    pub fn new(x: &'a [usize]) -> Self {
        let n = x.len();
        let block_size = block_size(n);

        let (reduced_idx, reduced_vals) = reduce_array(x, block_size);
        

        let _reduced = _ReducedBuilder {
            x, block_size, reduced_vals, reduced_idx,
            sparse_builder: |red_vals: &Vec<usize>| Sparse::new(&red_vals),
        }
        .build();
        Reduced(_reduced)
    }

    // accessors -- not public
    fn x(&self) -> &'a [usize] { return self.0.borrow_x() }
    fn block_size(&self) -> BlockSize { return *self.0.borrow_block_size() }
    fn reduced_idx(&self) -> &[usize] { return &self.0.borrow_reduced_idx() }
    fn sparse_rmq(&self, bi: BlockIdx, bj: BlockIdx) -> Option<usize> { 
        let (BlockIdx(i), BlockIdx(j)) = (bi, bj);
        return Some(self.reduced_idx()[self.0.borrow_sparse().rmq(i, j)?])
    }
}

impl<'a> RMQ for Reduced<'a> {
        fn rmq(&self, i: usize, j: usize) -> Option<usize> {
           let (bi, ii) = round_up(i, self.block_size());
           let (bj, jj) = round_down(j, self.block_size());
           if bi < bj {
                let p1 = Point::get(super::rmq(&self.x(), i, ii), self.x());
                let p2 = Point::get(self.sparse_rmq(bi, bj), self.x());
                let p3 = Point::get(super::rmq(&self.x(), jj, j), self.x());
                let min = super::lift_op(cmp::min);
                return Some(min(min(p1, p2), p3)?.0)
        } else {
                super::rmq(self.x(), i, j)
        }

    }
}