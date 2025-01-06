use pyo3::prelude::*;
use num_integer::binomial;

fn bernstein_poly_rust(n: u64, i: u64, t: f64) -> f64 {
    return (binomial(n, i) as f64) * t.powf(i as f64) * (1.0 - t).powf((n - i) as f64);
}

#[pyfunction]
fn bernstein_poly(n: u64, i: u64, t: f64) -> PyResult<f64> {
    Ok(bernstein_poly_rust(n, i, t))
}

#[pyfunction]
fn bezier_curve_eval(P: Vec<Vec<f64>>, t: f64) -> PyResult<Vec<f64>> {
    let n = P.len() - 1;
    let dim = P[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut b_poly: f64 = 0.0;
    for i in 0..n+1 {
        b_poly = bernstein_poly_rust(n as u64, i as u64, t);
        for j in 0..dim {
            evaluated_point[j] += P[i][j] * b_poly;
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn rational_bezier_curve_eval(P: Vec<Vec<f64>>, w: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let n = P.len() - 1;
    let dim = P[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    let mut b_poly: f64 = 0.0;
    for i in 0..n+1 {
        b_poly = bernstein_poly_rust(n as u64, i as u64, t);
        w_sum += w[i] * b_poly;
        for j in 0..dim {
            evaluated_point[j] += P[i][j] * b_poly;
        }
    }
    for j in 0..dim {
        evaluated_point[j] /= w_sum;
    }
    Ok(evaluated_point)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bernstein_poly, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_eval, m)?)?;
    Ok(())
}
