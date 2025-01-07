use pyo3::prelude::*;
use num_integer::binomial;

fn bernstein_poly_rust(n: usize, i: usize, t: f64) -> f64 {
    return (binomial(n, i) as f64) * t.powf(i as f64) * (1.0 - t).powf((n - i) as f64);
}

#[pyfunction]
fn bernstein_poly(n: usize, i: usize, t: f64) -> PyResult<f64> {
    Ok(bernstein_poly_rust(n, i, t))
}

#[pyfunction]
fn bezier_curve_eval(P: Vec<Vec<f64>>, t: f64) -> PyResult<Vec<f64>> {
    let n = P.len() - 1;
    let dim = P[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly = bernstein_poly_rust(n, i, t);
        for j in 0..dim {
            evaluated_point[j] += P[i][j] * b_poly;
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn bezier_surf_eval(P: Vec<Vec<Vec<f64>>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = P.len() - 1;  // Degree in the u-direction
    let m = P[0].len() - 1;  // Degree in the v-direction
    let dim = P[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            let b_poly_prod = b_poly_u * b_poly_v;
            for k in 0..dim {
                evaluated_point[k] += P[i][j][k] * b_poly_prod;
            }
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
    for i in 0..n+1 {
        let b_poly = bernstein_poly_rust(n, i, t);
        w_sum += w[i] * b_poly;
        for j in 0..dim {
            evaluated_point[j] += P[i][j] * w[i] * b_poly;
        }
    }
    for j in 0..dim {
        evaluated_point[j] /= w_sum;
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn rational_bezier_surf_eval(P: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let n = P.len() - 1;  // Degree in the u-direction
    let m = P[0].len() - 1;  // Degree in the v-direction
    let dim = P[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..n+1 {
        let b_poly_u = bernstein_poly_rust(n, i, u);
        for j in 0..m+1 {
            let b_poly_v = bernstein_poly_rust(m, j, v);
            let b_poly_prod = b_poly_u * b_poly_v;
            w_sum += w[i][j] * b_poly_prod;
            for k in 0..dim {
                evaluated_point[k] += P[i][j][k] * w[i][j] * b_poly_prod;
            }
        }
    }
    for k in 0..dim {
        evaluated_point[k] /= w_sum;
    }
    Ok(evaluated_point)
}

fn get_possible_span_indices(k: &[f64]) -> Vec<usize> {
    let mut possible_span_indices: Vec<usize> = Vec::new();
    let num_knots = k.len();
    for i in 0..num_knots-1 {
        if k[i] == k[i + 1] {
            continue;
        }
        possible_span_indices.push(i);
    }
    return possible_span_indices;
}

fn find_span(k: &[f64], possible_span_indices: &[usize], t: f64) -> usize {
    for &knot_span_idx in possible_span_indices {
        if k[knot_span_idx] <= t && t < k[knot_span_idx + 1] {
            return knot_span_idx;
        }
    }
    // If the parameter value is equal to the last knot, just return the last possible knot span index
    if t == k[k.len() - 1] {
        return possible_span_indices[possible_span_indices.len() - 1];
    }
    let k1: f64 = k[0];
    let k2: f64 = k[k.len() - 1];
    panic!("{}",
        format!("Parameter value t = {t} out of bounds for knot vector with first knot {k1} and last knot {k2}")
    );
}

fn cox_de_boor(k: &[f64], possible_span_indices: &[usize], p: usize, i: usize, t: f64) -> f64 {
    if p == 0 {
        if possible_span_indices.contains(&i) && find_span(&k, &possible_span_indices, t) == i {
            return 1.0;
        }
        return 0.0;
    }
    let mut f: f64 = 0.0;
    let mut g: f64 = 0.0;
    if k[i + p] - k[i] != 0.0 {
        f = (t - k[i]) / (k[i + p] - k[i]);
    }
    if k[i + p + 1] - k[i + 1] != 0.0 {
        g = (k[i + p + 1] - t) / (k[i + p + 1] - k[i + 1]);
    }
    if f == 0.0 && g == 0.0 {
        return 0.0;
    }
    if g == 0.0 {
        return f * cox_de_boor(&k, &possible_span_indices, p - 1, i, t);
    }
    if f == 0.0 {
        return g * cox_de_boor(&k, &possible_span_indices, p - 1, i + 1, t);
    }
    return f * cox_de_boor(&k, &possible_span_indices, p - 1, i, t) + g * cox_de_boor(
        &k, &possible_span_indices, p - 1, i + 1, t);
}

#[pyfunction]
fn bspline_curve_eval(P: Vec<Vec<f64>>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let num_cps = P.len();
    let num_knots = k.len();
    let degree = num_knots - num_cps - 1;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let n = P.len() - 1;
    let dim = P[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..n+1 {
        let bspline_basis = cox_de_boor(&k, &possible_span_indices, degree, i, t);
        for j in 0..dim {
            evaluated_point[j] += P[i][j] * bspline_basis;
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn bspline_surf_eval(P: Vec<Vec<Vec<f64>>>, ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let num_cps_u = P.len();  // Number of control points in the u-direction
    let num_cps_v = P[0].len();  // Number of control points in the v-direction
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let n = num_knots_u - num_cps_u - 1;  // Degree in the u-direction
    let m = num_knots_v - num_cps_v - 1;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = P[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    for i in 0..num_cps_u {
        let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, n, i, u);
        for j in 0..num_cps_v {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, m, j, v);
            let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
            for k in 0..dim {
                evaluated_point[k] += P[i][j][k] * bspline_basis_prod;
            }
        }
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn nurbs_curve_eval(P: Vec<Vec<f64>>, w: Vec<f64>, k: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
    let num_cps = P.len();
    let num_knots = k.len();
    let degree = num_knots - num_cps - 1;
    let possible_span_indices: Vec<usize> = get_possible_span_indices(&k);
    let n = P.len() - 1;
    let dim = P[0].len();
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..n+1 {
        let bspline_basis = cox_de_boor(&k, &possible_span_indices, degree, i, t);
        w_sum += w[i] * bspline_basis;
        for j in 0..dim {
            evaluated_point[j] += P[i][j] * w[i] * bspline_basis;
        }
    }
    for j in 0..dim {
        evaluated_point[j] /= w_sum;
    }
    Ok(evaluated_point)
}

#[pyfunction]
fn nurbs_surf_eval(P: Vec<Vec<Vec<f64>>>, w: Vec<Vec<f64>>,
    ku: Vec<f64>, kv: Vec<f64>, u: f64, v: f64) -> PyResult<Vec<f64>> {
    let num_cps_u = P.len();  // Number of control points in the u-direction
    let num_cps_v = P[0].len();  // Number of control points in the v-direction
    let num_knots_u = ku.len();  // Number of knots in the u-direction
    let num_knots_v = kv.len();  // Number of knots in the v-direction
    let n = num_knots_u - num_cps_u - 1;  // Degree in the u-direction
    let m = num_knots_v - num_cps_v - 1;  // Degree in the v-direction
    let possible_span_indices_u = get_possible_span_indices(&ku);
    let possible_span_indices_v = get_possible_span_indices(&kv);
    let dim = P[0][0].len();  // Number of spatial dimensions
    let mut evaluated_point: Vec<f64> = vec![0.0; dim];
    let mut w_sum: f64 = 0.0;
    for i in 0..num_cps_u {
        let bspline_basis_u = cox_de_boor(&ku, &possible_span_indices_u, n, i, u);
        for j in 0..num_cps_v {
            let bspline_basis_v = cox_de_boor(&kv, &possible_span_indices_v, m, j, v);
            let bspline_basis_prod = bspline_basis_u * bspline_basis_v;
            w_sum += w[i][j] * bspline_basis_prod;
            for k in 0..dim {
                evaluated_point[k] += P[i][j][k] * bspline_basis_prod;
            }
        }
    }
    for k in 0..dim {
        evaluated_point[k] /= w_sum;
    }
    Ok(evaluated_point)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bernstein_poly, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(rational_bezier_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_surf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_curve_eval, m)?)?;
    m.add_function(wrap_pyfunction!(nurbs_surf_eval, m)?)?;
    Ok(())
}
