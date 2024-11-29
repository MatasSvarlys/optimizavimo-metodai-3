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
    
    
    // println!("Starting at point ({}, {}, {})", point1.x, point1.y, point1.z);
    // let p1 = optimization_loop(point1.clone());
    // println!("\n\n");

    println!("Starting at point ({}, {}, {})", point2.x, point2.y, point2.z);
    let p2 = optimization_loop(point2.clone());
    println!("\n\n");

    // println!("Starting at point ({}, {}, {})", point3.x, point3.y, point3.z);
    // let p3 = optimization_loop(point3.clone());
    // println!("ans: {:?}, {:?}, {:?}", p1, p2, p3);
}

fn loss_function(p: Point, fault_coef: f64) -> f64{
    // p.print();

    let out = p.F()+(1.0/fault_coef)*p.B();

    println!("Loss function (F={}, b={}) value: {}", p.F(), p.B(), out);
    println!("{}", if p.B() != 0.0 { "Constraints breached!!" } else { "Fits constraints" });

    out
}

fn optimization_loop(mut p: Point) -> Point{
    let mut r = 100.0;
    let r_mul = 1.2;

    while calc_gradient_norm(p.gradient_B(r)) > 1e-4{
        //printing
        loss_function(p.clone(), r); 

        //calc next point
        p = linear_descent(p, r, r_mul);

        p.print_all();
        println!("Used constraint: {}\n", r);

        //update r for next loop
        r /= r_mul;
    }

    p
}