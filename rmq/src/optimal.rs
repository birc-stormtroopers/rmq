// Optimal <O(n),O(1)> solution using Cartesian trees and paths through ballot grids.

use super::reduce::{block_size, reduce_array, round_down, round_up, BlockIdx, BlockSize};
use super::{sparse::Sparse, tables::matrix, tabulate, Point, RMQ};

use ouroboros::self_referencing;
use std::cmp;

/// Build a table of Ballot numbers B_pq from p=q=0 to p=q=b.
fn tabulate_ballot_numbers(b: usize) -> matrix::Matrix {
    let mut ballot = matrix::Matrix::new(b + 1);
    for q in 0..=b {
        ballot[(0, q)] = 1
    }
    for q in 1..=b {
        for p in 1..=q {
            ballot[(p, q)] = ballot[(p - 1, q)] + ballot[(p, q - 1)]
        }
    }
    ballot
}

/// Compute the block type number of a block.
/// The b argument is the true block size, but it can differ from block.len() for the last
/// block. When we process the last block, we fake push the missing elements, putting them
/// lower in the Cartesian tree than the real ones, so we still get the right RMQ.
fn block_type(block: &[usize], b: usize, stack: &mut [i64], ballot: &matrix::Matrix) -> usize {
    let mut num = 0;
    let mut top = 0;
    stack[top] = i64::MIN; // As close to -infinity as we get with this type...

    for (i, &v) in block.iter().enumerate() {
        let signed_v = v as i64;

        // Invariant: When i runs from zero to b, b-i is p in B[p,q].
        //            i-top is how the depth of the stack and b-(i-top)
        //            is then the q in B[p,q].
        let p = b - i;
        while stack[top] > signed_v {
            // Popping
            let q = b - (i - top);
            num += ballot[(p - 1, q)];
            top -= 1;
        }

        // Push...
        top += 1;
        stack[top] = signed_v;
    }

    return num;
}

/// Compute the block types for all blocks in x and compute the tables for the
/// blocks we observe.
fn tabulate_blocks(x: &[usize], b: usize) -> (Vec<usize>, Vec<Option<tabulate::TabulatedQuery>>) {
    // We need to round up to get the number of blocks here.
    // The reduced array handles blocks 0, 1, ..., x.len()/b but we
    // might also have a block after it.
    let no_blocks = (x.len() + b - 1) / b;

    let ballot = tabulate_ballot_numbers(b);
    let mut stack: Vec<i64> = vec![i64::MIN; b + 1];

    let mut block_types = vec![0; no_blocks];
    let mut block_tables = vec![None; ballot[(b, b)]];
    for i in 0..no_blocks {
        let begin = i * b;
        // The last block might be smaller than a full block, but if we just
        // tabulate it anyway the missing values are virtually pushed and behave
        // like they are larger than the existing ones, giving us the right RMQ
        // results anyway (the true values are always smaller than the virtual ones).
        let end = cmp::min(x.len(), begin + b);
        let block = &x[begin..end];

        let bt = block_type(block, b, &mut stack, &ballot);
        block_types[i] = bt;
        if let None = block_tables[bt] {
            block_tables[bt] = Some(tabulate::TabulatedQuery::new(block));
        }
    }
    return (block_types, block_tables);
}

#[self_referencing]
struct _Optimal<'a> {
    // Original data
    x: &'a [usize],

    // Reduced table
    block_size: BlockSize,
    reduced_vals: Vec<usize>,
    reduced_idx: Vec<usize>,
    #[borrows(reduced_vals)]
    #[covariant]
    sparse: Sparse<'this>,

    // Block types and tables
    block_types: Vec<usize>,
    block_tables: Vec<Option<tabulate::TabulatedQuery>>,
}
pub struct Optimal<'a>(_Optimal<'a>);

impl<'a> Optimal<'a> {
    #[allow(dead_code)] // only used in tests right now
    pub fn new(x: &'a [usize]) -> Self {
        let n = x.len();
        let BlockSize(b) = block_size(n);

        // adjust block size; log(n) is too much so change it to a quarter.
        let b = cmp::max(4, b / 4); // I don't want too small blocks, so minimum is 4
        let block_size = BlockSize(b);

        let (reduced_idx, reduced_vals) = reduce_array(x, block_size);
        let (block_types, block_tables) = tabulate_blocks(x, b);

        let _optimal = _OptimalBuilder {
            x,
            block_size,
            reduced_vals,
            reduced_idx,
            sparse_builder: |red_vals: &Vec<usize>| Sparse::new(&red_vals),
            block_types,
            block_tables,
        }
        .build();
        Optimal(_optimal)
    }

    // accessors -- not public
    fn x(&self) -> &'a [usize] {
        return self.0.borrow_x();
    }
    fn block_size(&self) -> BlockSize {
        return *self.0.borrow_block_size();
    }
    fn reduced_idx(&self) -> &[usize] {
        return &self.0.borrow_reduced_idx();
    }
    fn sparse_rmq(&self, bi: BlockIdx, bj: BlockIdx) -> Option<usize> {
        let (BlockIdx(i), BlockIdx(j)) = (bi, bj);
        return Some(self.reduced_idx()[self.0.borrow_sparse().rmq(i, j)?]);
    }

    fn block_rmq(&self, i: usize, j: usize) -> Option<Point> {
        if i < j {
            // Get misc values and tables we need...
            let BlockSize(bs) = self.block_size();
            let block_index = i / bs; // The index in the list of blocks
            let block_begin = block_index * bs; // The index the block starts at in x

            let block_types = self.0.borrow_block_types();
            let block_tables = self.0.borrow_block_tables();

            // Get the table for this block by looking up the block type and then the
            // table from the block type.
            let tbl = block_tables[block_types[block_index]].as_ref().unwrap();

            // Get RMQ and adjust the index back up, so it is relative to the start of the block.
            let rmq_idx = Some(tbl.rmq(i - block_begin, j - block_begin)? + block_begin);
            Point::get(rmq_idx, self.x())
        } else {
            // j <= i so not a valid interval.
            None
        }
    }
}

impl<'a> RMQ for Optimal<'a> {
    fn rmq(&self, i: usize, j: usize) -> Option<usize> {
        let BlockSize(bs) = self.block_size();
        // The block indices are not the same for the small tables and the
        // sparse table. For the sparse table we have to round up for i, but
        // to get the block i is in, we need to round down.
        let bi = BlockIdx(i / bs);
        let (sparse_bi, ii) = round_up(i, BlockSize(bs));
        let (bj, jj) = round_down(j, BlockSize(bs));

        if bi < bj {
            let p1 = self.block_rmq(i, ii);
            let p2 = Point::get(self.sparse_rmq(sparse_bi, bj), self.x());
            let p3 = self.block_rmq(jj, j);
            let min = super::lift_op(cmp::min);
            Some(min(min(p1, p2), p3)?.0)
        } else {
            Some(self.block_rmq(i, j)?.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumeration() {
        let b = 4;
        let ballot = tabulate_ballot_numbers(b);
        let mut stack: Vec<i64> = vec![i64::MIN; b + 1];

        let v = vec![1, 2, 3, 4]; // all push
        assert_eq!(0, block_type(&v, b, &mut stack, &ballot));

        let v = vec![4, 2, 3, 1];
        assert_eq!(9 + 1 + 1, block_type(&v, b, &mut stack, &ballot));

        let v = vec![4, 3, 2, 1]; // all pop
        assert_eq!(ballot[(b, b)] - 1, block_type(&v, b, &mut stack, &ballot));

        let v = vec![1, 2, 3, 4, 4, 2, 3, 1, 4, 3, 2, 1];
        let (block_types, _) = tabulate_blocks(&v, b);
        assert_eq!(block_types.len(), 3);
        assert_eq!(block_types[0], 0);
        assert_eq!(block_types[1], 9 + 1 + 1);
        assert_eq!(block_types[2], ballot[(b, b)] - 1);

        // last block has length 1; it will work as an "all push"
        let v = vec![1, 2, 3, 4, 4, 2, 3, 1, 4, 3, 2, 1, 42];
        let (block_types, _) = tabulate_blocks(&v, b);
        assert_eq!(block_types.len(), 4);
        assert_eq!(block_types[0], 0);
        assert_eq!(block_types[1], 9 + 1 + 1);
        assert_eq!(block_types[2], ballot[(b, b)] - 1);
        assert_eq!(block_types[3], 0);

        // last block has length 2; it will work as a push and a pop (for 9)
        let v = vec![1, 2, 3, 4, 4, 2, 3, 1, 4, 3, 2, 1, 42, 21];
        let (block_types, _) = tabulate_blocks(&v, b);
        assert_eq!(block_types.len(), 4);
        assert_eq!(block_types[0], 0);
        assert_eq!(block_types[1], 9 + 1 + 1);
        assert_eq!(block_types[2], ballot[(b, b)] - 1);
        assert_eq!(block_types[3], 9);
    }

    #[test]
    fn test_tables() {
        let b = 4;

        // last block has length 2; it will work as a push and a pop (for 9)
        let v = vec![1, 2, 3, 4, 4, 2, 3, 1, 4, 3, 2, 1, 42, 21];
        let (block_types, block_tables) = tabulate_blocks(&v, b);

        // block 1, 2, 3, 4
        let tbl = block_tables[block_types[0]].as_ref().unwrap();
        assert_eq!(tbl.rmq(0, 1), Some(0));
        assert_eq!(tbl.rmq(0, 2), Some(0));
        assert_eq!(tbl.rmq(0, 3), Some(0));
        assert_eq!(tbl.rmq(0, 4), Some(0));
        assert_eq!(tbl.rmq(1, 2), Some(1));
        assert_eq!(tbl.rmq(1, 3), Some(1));
        assert_eq!(tbl.rmq(1, 4), Some(1));
        assert_eq!(tbl.rmq(2, 3), Some(2));
        assert_eq!(tbl.rmq(2, 4), Some(2));
        assert_eq!(tbl.rmq(3, 4), Some(3));

        // block 4, 2, 3, 1
        let tbl = block_tables[block_types[1]].as_ref().unwrap();
        assert_eq!(tbl.rmq(0, 1), Some(0));
        assert_eq!(tbl.rmq(0, 2), Some(1));
        assert_eq!(tbl.rmq(0, 3), Some(1));
        assert_eq!(tbl.rmq(0, 4), Some(3));
        assert_eq!(tbl.rmq(1, 2), Some(1));
        assert_eq!(tbl.rmq(1, 3), Some(1));
        assert_eq!(tbl.rmq(1, 4), Some(3));
        assert_eq!(tbl.rmq(2, 3), Some(2));
        assert_eq!(tbl.rmq(2, 4), Some(3));
        assert_eq!(tbl.rmq(3, 4), Some(3));

        // block 4, 3, 2, 1
        let tbl = block_tables[block_types[2]].as_ref().unwrap();
        assert_eq!(tbl.rmq(0, 1), Some(0));
        assert_eq!(tbl.rmq(0, 2), Some(1));
        assert_eq!(tbl.rmq(0, 3), Some(2));
        assert_eq!(tbl.rmq(0, 4), Some(3));
        assert_eq!(tbl.rmq(1, 2), Some(1));
        assert_eq!(tbl.rmq(1, 3), Some(2));
        assert_eq!(tbl.rmq(1, 4), Some(3));
        assert_eq!(tbl.rmq(2, 3), Some(2));
        assert_eq!(tbl.rmq(2, 4), Some(3));
        assert_eq!(tbl.rmq(3, 4), Some(3));

        // block 42, 21, -, -
        let tbl = block_tables[block_types[3]].as_ref().unwrap();
        assert_eq!(tbl.rmq(0, 1), Some(0));
        assert_eq!(tbl.rmq(0, 2), Some(1));
        assert_eq!(tbl.rmq(1, 2), Some(1));
    }
}
