/// Upper-triangular tables
pub mod triag {
    #[inline]
    fn flat_index(i: usize, j: usize, n: usize) -> usize {
        let k = n - i - 1;
        k * (k + 1) / 2 + j - i - 1
    }
    
    /// Table for looking up at [i,j) (j > i) intervals.
    pub struct UTTable {
        n: usize,
        table: Vec<usize>,
    }
    
    impl UTTable {
        pub fn new(n: usize) -> UTTable {
            let table: Vec<usize> = vec![0; n * (n + 1) / 2];
            UTTable { n, table }
        }
    }
    
    impl std::ops::Index<(usize, usize)> for UTTable {
        type Output = usize;
        fn index(&self, index: (usize, usize)) -> &Self::Output {
            let (i, j) = index;
            assert!(i < self.n);
            assert!(i < j && j <= self.n);
            &self.table[flat_index(i, j, self.n)]
        }
    }
    
    impl std::ops::IndexMut<(usize, usize)> for UTTable {
        fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
            let (i, j) = index;
            assert!(i < self.n);
            assert!(i < j && j <= self.n);
            &mut self.table[flat_index(i, j, self.n)]
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_triag() {
            let n = 5;
            let mut tbl = UTTable::new(n);
            for i in 0..n {
                for j in (i+1)..n {
                    tbl[(i,j)] = j;
                }
            }
            for i in 0..n {
                for j in (i+1)..n {
                    assert!(j == tbl[(i,j)]);
                }
            }
        }
    }
}

