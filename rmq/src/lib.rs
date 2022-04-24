pub trait RMQ<'a> {
    fn new(x: &'a [usize]) -> Self;
    fn rmq(&self, i: usize, j: usize) -> usize;
}

// Linear query time with zero preprocessing: <O(1),O(n)>
pub struct LinearQuery<'a> {
    x: &'a [usize],
}

impl<'a> RMQ<'a> for LinearQuery<'a> {
    fn new(x: &'a [usize]) -> Self {
        LinearQuery{x: &x}
    }
    fn rmq(&self, i: usize, j: usize) -> usize {
        let y = &self.x[i..j];
        let min_val = y.iter().min().unwrap();
        let pos = i + y.iter().position(|a| a == min_val).unwrap();
        pos
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn check_min_in_interval<'a, R: RMQ<'a>>(x: &'a [usize], rmq: &'a R, i: usize, j: usize) {
        let k = rmq.rmq(i, j);
        assert!(i <= k);
        assert!(k < j);

        let v = x[k];
        for l in i..k {
            assert!(x[l] > v);
        }
        for l in k + 1..j {
            assert!(x[l] >= v)
        }
    }

    fn check_min<'a, R: 'a + RMQ<'a>>(x: &'a [usize], rmq: &'a R) {
        for i in 0..x.len() {
            for j in i + 1..x.len() + 1 {
                check_min_in_interval(x, rmq, i, j)
            }
        }
    }

    #[test]
    fn test_linear() {
        // Not power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4];
        check_min(&x, &LinearQuery::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4, 2, 6, 3, 4, 7, 9];
        check_min(&x, &LinearQuery::new(&x));
        // Not power of two
        let x = vec![2, 1, 2, 0, 2, 1, 3, 7, 4];
        check_min(&x, &LinearQuery::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3];
        check_min(&x, &LinearQuery::new(&x));
    }
}
