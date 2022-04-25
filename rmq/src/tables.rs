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

/// Storing values for all (i,i+2^k) indices and only those
pub mod powers {
    /// Type for powers of two, 2^k. Contains k, but wrapped in
    /// a type so we don't confuse log-space with linear space.
    #[derive(Debug, Clone, Copy)]
    pub struct Pow(pub usize);
    
    impl std::cmp::PartialEq for Pow {
        #[inline]
        fn eq(&self, other: &Pow) -> bool {
            self.0 == other.0
        }
    }
    
    impl std::cmp::PartialOrd for Pow {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            let Pow(i) = *self;
            let Pow(j) = *other;
            Some(i.cmp(&j))
        }
    }
    
    impl std::fmt::Display for Pow {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "2^{}", self.0)
        }
    }
    
    impl Pow {
        /// for a power Pow(k) get 2^k.
        #[inline]
        pub fn value(&self) -> usize {
            1 << self.0
        }
    }
    
    /// Get k such that 2**k is j rounded down to the
    /// nearest power of 2.
    /// j=1=2^0 => 0
    /// j=2=2^1 => 1
    /// j=3=2^1+1 => 1
    /// j=4=2^2 => 2
    /// and so on.
    pub fn log2_down(j: usize) -> Pow {
        assert!(j != 0); // not defined for zero
        
        // Rounded down means finding the index of the first
        // 1 in the bit-pattern. If j = 00010101110
        // then 00010000000 (only first bit) is the closest
        // power of two, and we want the position of that bit.
        // j.leading_zeros() counts the number of leading zeros
        // and we get the index by subtracting this
        // from the total number of bits minus one.
        Pow((usize::BITS - j.leading_zeros() - 1) as usize)
        // usize::BITS and j.leading_zeros() will be u32, so
        // we cast the result back to usize.
    }

    pub fn power_of_two(x: usize) -> bool {
        (x == 0) || ((x & (x - 1)) == 0)
    }

    /// For n, get (rounded up) log2(n).
    pub fn log2_up(n: usize) -> Pow {
        // log_table_size(n) with n=2^k+m will always give us 2^{k+1},
        // whether m is zero or not. We want 2^{k+1} when m > 0 and 2^k
        // when m is zero, i.e. when n is a power of two.
        // So we should subtract one from the exponent if n is a power of two.
        let Pow(k) = log_table_size(n);
        Pow(k - power_of_two(n) as usize)
    }
    
    /// We always have to add one to the exponent, because in log-space
    /// we are working with 1-indexed (0-indexed in log-space) values,
    /// so to have a table that can handle maximum value k, we need k+1
    /// entires. That is what this function gives us.
    pub fn log_table_size(n: usize) -> Pow {
        let Pow(k) = log2_down(n);
        Pow(k + 1)
    }
    
    /// From range [i,j), get values (k,j-2^k) where k is the offset
    /// into the TwoD table to look up the value for [i,i+2^k) and [j-2^k,j)
    /// from which we can get the RMQ.
    pub fn power_index(i: usize, j: usize) -> ((usize,Pow), (usize,Pow)) {
        let powk = log2_down(j - i);
        (
            (i, powk),
            (j - powk.value(), powk)
        )
    }
    
    /// A rather simple 2D array made from vectors of vectors.
    /// There are better solutions, but I can implement those later
    /// with the same interface.
    pub struct Powers {
        table: Vec<Vec<usize>>,
    }
    
    impl Powers {
        pub fn new(n: usize) -> Powers {
            let Pow(logn) = log_table_size(n);
            let table = vec![vec![0; logn]; n];
            Powers { table }
        }
    }
    
    impl std::ops::Index<(usize, Pow)> for Powers {
        type Output = usize;
        fn index(&self, index: (usize, Pow)) -> &Self::Output {
            let (i, Pow(k)) = index;
            &self.table[i][k]
        }
    }
    
    impl std::ops::IndexMut<(usize, Pow)> for Powers {
        fn index_mut(&mut self, index: (usize, Pow)) -> &mut Self::Output {
            let (i, Pow(k)) = index;
            &mut self.table[i][k]
        }
    }
    
    impl std::fmt::Display for Powers {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            for row in &self.table {
                for val in row {
                    let _ = write!(f, "{} ", val);
                }
                let _ = write!(f, "\n");
            }
            Ok(())
        }
    }
    
    #[cfg(test)]
    mod tests {
        use super::*;
        
        #[test]
        fn test_power_index() {
            // [0,0) is undefined; j must be larger than i
            // [0,1) => offset=1=2^0, k=0, jj=1-2^k=0
            let ((ii, Pow(k)), (jj, Pow(kk))) = power_index(0, 1);
            assert_eq!(ii, 0);
            assert_eq!(k, kk);
            assert_eq!(k, 0);
            assert_eq!(jj, 0);
            
            // [0,2) => offset=2=2^1, k=1, jj=2-2^1=0
            let ((ii,Pow(k)), (jj, Pow(kk))) = power_index(0, 2);
            assert_eq!(0, ii);
            assert_eq!(k, kk);
            assert_eq!(k, 1);
            assert_eq!(jj, 0);
            
            // [0,3) => offset=2, k=1 -- second offset; then jj=1
            let ((ii,Pow(k)), (jj, Pow(kk))) = power_index(0, 3);
            assert_eq!(ii, 0);
            assert_eq!(k, kk);
            assert_eq!(k, 1);
            assert_eq!(jj, 1);
            
            // [0,4) => offset=4=2^2, k=2, jj=4-4=0
            let ((ii,Pow(k)), (jj, Pow(kk))) = power_index(0, 4);
            assert_eq!(ii, 0);
            assert_eq!(k, kk);
            assert_eq!(k, 2);
            assert_eq!(jj, 0);
            
            let ((ii,Pow(k)), (jj,_)) = power_index(0, 5);
            assert_eq!(ii, 0);
            assert_eq!(k, 2);
            assert_eq!(jj, 1);
            
            let ((ii,Pow(k)), (jj, _)) = power_index(0, 6);
            assert_eq!(ii, 0);
            assert_eq!(k, 2);
            assert_eq!(jj, 2);
            
            let ((_,Pow(k)), (jj, _)) = power_index(0, 7);
            assert_eq!(k, 2);
            assert_eq!(jj, 3);
            
            let ((_,Pow(k)), (jj, _)) = power_index(0, 8);
            assert_eq!(k, 3);
            assert_eq!(jj, 0);
            
            let ((_,Pow(k)), (jj, _)) = power_index(1, 8);
            assert_eq!(k, 2);
            assert_eq!(jj, 4);
            
            let ((_,Pow(k)), (jj, _)) = power_index(1, 9);
            assert_eq!(k, 3);
            assert_eq!(jj, 1);
        }
        
        #[test]
        fn test_2d() {
            let n = 5;
            let mut tbl = Powers::new(n);
            println!("{}", tbl);
            
            for i in 0..n {
                for j in i + 1..n + 1 {
                    let ((_,Pow(k)), _) = power_index(i, j);
                    assert_eq!(0, tbl[(i, Pow(k))]);
                }
            }
            
            for i in 0..n {
                for j in i + 1..n + 1 {
                    // This is just saving the largest matching j
                    // in the entry those js should go to
                    let ((_,Pow(k)), _) = power_index(i, j);
                    println!("({},{}) to offset {}", i, j, k);
                    tbl[(i, Pow(k))] = j;
                }
            }
            println!("{}", tbl);
            for i in 0..n {
                for j in i + 1..n + 1 {
                    let ((_,Pow(k)), _) = power_index(i, j);
                    println!("({},{}): {} <? {}", i, j, j, tbl[(i, Pow(k))]);
                    assert!(j <= tbl[(i, Pow(k))]);
                }
            }
        }
    }
}
