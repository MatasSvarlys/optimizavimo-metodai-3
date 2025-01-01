use base_funcs::{calc_gradient_norm, check_value, Point};
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
    
    let r_vals = vec![500.0, 50.0, 5.0];
    let r_mul_vals = vec![5.0, 10.0, 100.0, 1000.0, 10000.0];

    // println!("Starting at point ({}, {}, {})", point1.x, point1.y, point1.z);
    // for r in r_vals.iter(){
    //     for r_mul in r_mul_vals.iter(){
    //         optimization_loop(point1.clone(), *r, *r_mul);
    //     }
    // }
    
    // println!("Starting at point ({}, {}, {})", point2.x, point2.y, point2.z);
    // for r in r_vals.iter(){
    //     for r_mul in r_mul_vals.iter(){
    //         optimization_loop(point2.clone(), *r, *r_mul);
    //     }
    // }

    println!("Starting at point ({}, {}, {})", point3.x, point3.y, point3.z);
    for r in r_vals.iter(){
        for r_mul in r_mul_vals.iter(){
            optimization_loop(point3.clone(), *r, *r_mul);
        }
    }
}

fn loss_function(p: Point, fault_coef: f64) -> f64{
    let out = p.F()+(1.0/fault_coef)*p.B();

    out
}

fn optimization_loop(mut p: Point, mut r: f64, r_mul: f64){
    let mut gama = 0.8; //Step size
    let mut loop_count = 0;

    println!("r gets divided by: {}", r_mul);
    println!("| Loop |    r     | Step Size |     x     |     y     |     z     |    F    |    B    | Steps Taken | Loss Func Eval |");
    println!("----------------------------------------------------------------------------------------------------------------------");

    //Initial point
    let F = p.F();
    let B = p.B();
    let loss_eval = loss_function(p.clone(), r); //Initial loss function evaluation
    println!(
        "| {:<4} | {:<8.3} | {:<9.3} | {:<9.3} | {:<9.3} | {:<9.3} | {:<7.3} | {:<7.3} | {:<11} | {:<14.3} |",
        loop_count, r, gama, p.x, p.y, p.z, F, B, "-", loss_eval
    );

    while calc_gradient_norm(p.gradient_B(r)) > 1e-4 {
        loop_count += 1;
        let mut step_count: i32;

        //Calculate next point
        (p, step_count) = linear_descent(p, r, gama);
        
        if let Err(e) = check_value(p.G1()) {
            eprintln!("{}", e);
            break;
        }

        let F = p.F();
        let B = p.B();
        let loss_eval = loss_function(p.clone(), r);

        if F.is_nan() || B.is_nan() || loss_eval.is_nan() {
           println!("G1 got out of bounds");
           break;
        }

        println!(
            "| {:<4} | {:<8.5} | {:<9.6} | {:<9.6} | {:<9.6} | {:<9.6} | {:<7.4} | {:<7.3} | {:<11} | {:<14.9} |",
            loop_count, r, gama, p.x, p.y, p.z, F, B, step_count, loss_eval
        );

        //Update r and gama for the next loop
        gama /= r_mul;
        let min_r = 1e-2; //Adjust as needed
        r = f64::max(r / r_mul, min_r);
    }
    println!("\n");
}
