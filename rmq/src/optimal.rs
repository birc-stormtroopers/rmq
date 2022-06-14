// Optimal <O(n),O(1)> solution using Cartesian trees and paths through ballot grids.

mod number {
    use crate::tables::matrix;

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

    fn block_type(block: &[usize], stack: &mut [i64], ballot: &matrix::Matrix) -> usize {
        let b = block.len();
        let mut num = 0;
        let mut top = 0;
        stack[top] = i64::MIN;

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

        return num
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_enumeration() {
            let v = vec![8, 3, 6, 1];

            let b = v.len();
            let ballot = tabulate_ballot_numbers(b);
            let mut stack: Vec<i64> = vec![i64::MIN; b + 1];

            // FIXME: handle last block if v.len() is not a multiple of b
            let no_blocks = v.len() / b;
            for b_i in 0..no_blocks {
                let begin = b_i * b;
                let end = begin + b;
                let block = &v[begin..end];

                println!("block [{}..{}]", begin, end);
                assert_eq!(9 + 1 + 1, block_type(&block, &mut stack, &ballot));
            }
        }
    }
}
