use base_funcs::Point;

use crate::base_funcs::{self, calc_gradient_norm};

pub fn linear_descent(p: Point, r: f64, r_mul: f64) -> Point{
    let mut local_point = p.clone();
    let mut gradient = p.gradient_B(r);
    let mut gradient_norm = calc_gradient_norm(gradient.clone());
    let mut step_count = 1;
    
    //constants
    let gama = 0.8*r_mul; //step size
    let gradient_tolerance = 1e-4;
    
    //rest of steps
    // println!("{:?}", gradient);
    while gradient_norm > gradient_tolerance {
        //move towards the gradient cus we want to maximize the func
        local_point = local_point.move_towards_gradient(gradient.clone(), -gama);
        // local_point.print_all();

        //calc things for next loop
        gradient = local_point.gradient_B(r);
        gradient_norm = calc_gradient_norm(gradient.clone());
        
        step_count+=1;
    }

    println!("step count: {}", step_count);
    return local_point
}