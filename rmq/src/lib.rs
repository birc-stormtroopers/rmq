mod tables;
mod linear_query;


/// Range Minimum Query interface. Returns the left-most index
/// containing the minimum value in x[i:j] where x is the range
/// the RMQ preprocessing is done over. It returns an Option
/// since you sometimes don't want to check for empty intervals
/// before calling it, but instead want to deal with it after.
pub trait RMQ {
    fn rmq(&self, i: usize, j: usize) -> Option<usize>;
}




#[cfg(test)]
mod tests {
    use super::*;
    use super::linear_query::LinearQuery;

    fn check_min_in_interval<'a, R: RMQ>(x: &'a [usize], rmq: &'a R, i: usize, j: usize) {
        let k = rmq.rmq(i, j).unwrap();
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

    fn check_min<'a, R: RMQ>(x: &'a [usize], rmq: &'a R) {
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
