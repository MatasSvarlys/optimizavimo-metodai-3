use base_funcs::{calc_gradient_norm, Point};
use lab2::linear_descent;

mod base_funcs;
mod lab2;

fn main() {
    let point1 = Point{x:0.0, y:0.0, z:0.0};
    let point2 = Point{x:1.0, y:1.0, z:1.0};
    let point3 = Point{x:0.8, y:0.0, z:0.3};
    
    point1.print_all();
    point2.print_all();
    point3.print_all();
    println!("\n\n");
    
    
    println!("Starting at point ({}, {}, {})", point1.x, point1.y, point1.z);
    let p1 = optimization_loop(point1.clone());
    println!("\n\n");

    println!("Starting at point ({}, {}, {})", point2.x, point2.y, point2.z);
    let p2 = optimization_loop(point2.clone());
    println!("\n\n");

    println!("Starting at point ({}, {}, {})", point3.x, point3.y, point3.z);
    let p3 = optimization_loop(point3.clone());

    // println!("ans: {:?}, {:?}, {:?}", p1, p2, p3);
}

//
fn loss_function(p: Point, fault_coef: f64) -> f64{
    let out = p.F()+(1.0/fault_coef)*p.B();

    out
}
fn optimization_loop(mut p: Point) -> Point {
    let mut r = 50.0;
    let r_mul = 5.0;
    let mut gama = 0.8; // Step size
    let mut loop_count = 0;

    println!("r gets divided by: {}", r_mul);
    println!("| Loop |    r     | Step Size |     x     |     y     |     z     |    F    |    B    | Steps Taken | Loss Func Eval |");
    println!("----------------------------------------------------------------------------------------------------------------------");

    // Initial point
    let F = p.F();
    let B = p.B();
    let loss_eval = loss_function(p.clone(), r); // Initial loss function evaluation
    println!(
        "| {:<4} | {:<8.3} | {:<9.3} | {:<9.3} | {:<9.3} | {:<9.3} | {:<7.3} | {:<7.3} | {:<11} | {:<14.3} |",
        loop_count, r, gama, p.x, p.y, p.z, F, B, "-", loss_eval
    );

    while calc_gradient_norm(p.gradient_B(r)) > 1e-4 {
        loop_count += 1;
        let mut step_count: i32;

        // Calculate next point
        (p, step_count) = linear_descent(p, r, gama);

        let F = p.F();
        let B = p.B();
        let loss_eval = loss_function(p.clone(), r);

        println!(
            "| {:<4} | {:<8.5} | {:<9.6} | {:<9.6} | {:<9.6} | {:<9.6} | {:<7.4} | {:<7.3} | {:<11} | {:<14.9} |",
            loop_count, r, gama, p.x, p.y, p.z, F, B, step_count, loss_eval
        );

        // Update r and gama for the next loop
        gama /= r_mul;
        let min_r = 1e-2; // Adjust as needed
        r = f64::max(r / r_mul, min_r);
    }

    p
}
