use std::cmp;

mod tables;
mod linear_query;
mod tabulate;
mod sparse;
mod reduce;

/*
fn lift<A, B, C>(f: impl Fn(A, B)->C) -> impl Fn(Option<A>, Option<B>)->Option<C> {
    move |a, b| Some(f(a?, b?))
}
*/
fn lift_op<T: Copy>(f: impl Fn(T, T)->T) -> impl Fn(Option<T>, Option<T>)->Option<T> {
    move |a, b|         
    match (a, b) {
        (None, None) => None,
        (Some(_), None) => a,
        (None, Some(_)) => b,
        (Some(a), Some(b)) => Some(f(a,b)),
    }
}


fn rmq(x: &[usize], i: usize, j: usize) -> Option<usize> {
    let y = &x[i..j];
    let min_val = y.iter().min()?;
    let pos = i + y.iter().position(|a| a == min_val)?;
    Some(pos)
}

/// Range Minimum Query interface. Returns the left-most index
/// containing the minimum value in x[i:j] where x is the range
/// the RMQ preprocessing is done over. It returns an Option
/// since you sometimes don't want to check for empty intervals
/// before calling it, but instead want to deal with it after.
pub trait RMQ {
    fn rmq(&self, i: usize, j: usize) -> Option<usize>;
}

/// An index i together with its value x[i].
#[derive(Clone, Copy, Debug)]
pub struct Point(usize, usize);
impl Point {
    #[inline]
    pub fn new(i: usize, x: &[usize]) -> Point {
        Point(i, x[i])
    }
    /// Get Some(Point(i,x[i])) if the index is valid or None if not
    #[inline]
    pub fn get(i: Option<usize>, x: &[usize]) -> Option<Point> {
        Some(Point(i?, *x.get(i?)?))
    }

}
impl cmp::PartialEq for Point {
    fn eq(&self, other: &Point) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl cmp::Eq for Point {}
impl cmp::Ord for Point {
        fn cmp(&self, other: &Point) -> cmp::Ordering {
        if *self == *other {
            cmp::Ordering::Equal
        }
        else if self.1 < other.1 || (self.1 == other.1 && self.0 < other.0) {
            cmp::Ordering::Less
        }
        else {
            cmp::Ordering::Greater
        }
    }
}
impl cmp::PartialOrd for Point {
    fn partial_cmp(&self, other: &Point) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use super::linear_query::LinearQuery;
    use super::tabulate::TabulatedQuery;
    use super::sparse::Sparse;
    use super::reduce::Reduced;

    fn check_min_in_interval<'a, R: RMQ>(x: &'a [usize], rmq: &'a R, i: usize, j: usize) {
        let k = rmq.rmq(i, j).unwrap();
        println!("RMQ({},{}) = {}", i, j, k);
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

    #[test]
    fn test_tabulate() {
        // Not power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4];
        check_min(&x, &TabulatedQuery::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4, 2, 6, 3, 4, 7, 9];
        check_min(&x, &TabulatedQuery::new(&x));
        // Not power of two
        let x = vec![2, 1, 2, 0, 2, 1, 3, 7, 4];
        check_min(&x, &TabulatedQuery::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3];
        check_min(&x, &TabulatedQuery::new(&x));
    }

    #[test]
    fn test_sparse() {
        // Not power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4];
        check_min(&x, &Sparse::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4, 2, 6, 3, 4, 7, 9];
        check_min(&x, &Sparse::new(&x));
        // Not power of two
        let x = vec![2, 1, 2, 0, 2, 1, 3, 7, 4];
        check_min(&x, &Sparse::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3];
        check_min(&x, &Sparse::new(&x));
    }

    #[test]
    fn test_reduced() {
        // Not power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4];
        check_min(&x, &Reduced::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3, 7, 4, 2, 6, 3, 4, 7, 9];
        check_min(&x, &Reduced::new(&x));
        // Not power of two
        let x = vec![2, 1, 2, 0, 2, 1, 3, 7, 4];
        check_min(&x, &Reduced::new(&x));
        // Power of two
        let x = vec![2, 1, 2, 5, 3, 6, 1, 3];
        check_min(&x, &Reduced::new(&x));
    }
}
