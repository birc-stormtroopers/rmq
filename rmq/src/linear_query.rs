use super::RMQ;

/// Linear query time with zero preprocessing: <O(1),O(n)>
pub struct LinearQuery<'a> {
    x: &'a [usize],
}

impl<'a> LinearQuery<'a> {
    #[allow(dead_code)] // only used in tests right now
    pub fn new(x: &'a [usize]) -> Self {
        LinearQuery{x: &x}
    }
}

impl<'a> RMQ for LinearQuery<'a> {
    fn rmq(&self, i: usize, j: usize) -> Option<usize> {
        // The naive solution we already have in super:: as a function
        super::rmq(self.x, i, j)
    }
}
