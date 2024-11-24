use base_funcs::Point;

mod base_funcs;

fn main() {
    let point1 = Point{x:0.0, y:0.0, z:0.0};
    let point2 = Point{x:1.0, y:1.0, z:1.0};
    let point3 = Point{x:0.0, y:0.0, z:0.0};
    
    print_value_at_point(point1.clone());
    print_value_at_point(point2.clone());
    print_value_at_point(point3.clone());

    // let p1 = optimization_loop(point1.clone());
    let p2 = optimization_loop(point2.clone());
    // let p3 = optimization_loop(point3.clone());

    // println!("ans: {:?}, {:?}, {:?}", p1, p2, p3);
}


fn print_value_at_point(p: Point){
    println!("Sides: ({}, {}, {}); F={}; G={}; Hi=(1={}, 2={}, 3={})", p.x, p.y, p.z, p.F(), p.G1(), p.H1(), p.H2(), p.H3());
}

fn fault_function(p: Point, fault_coef: f64) -> f64{
    let sum_G_squared = p.G1().powi(2);    
    let max_H_squared = f64::max(p.H1(), f64::max(p.H2(), p.H3())).powi(2);
    let out = p.F()+(1.0/fault_coef)*(sum_G_squared+max_H_squared);
    // println!("called fault func (F={}, g^2={}, h^2={}) and got answ: {}", p.F(), sum_G_squared, max_H_squared, out);

    out
}

fn optimization_loop(mut p: Point) -> Point{
    let r_values = vec![10.0, 5.0, 1.0, 0.1];

    for i in r_values{
        println!("Current best values: ({}, {}, {}); fault at point: {}; Using fault coef: {}", p.x, p.y, p.z, fault_function(p.clone(), i), i);
        gradient_descent(&mut p, i);
    }
    println!("Ended on point: {:?}", p);
    p
}

fn gradient_descent(p: &mut Point, r: f64) -> Point{
    let step_size = 0.1;
    let tolerance = 10e-8;
    let mut gradient = calc_gradient(p.clone(), r);
    let mut iter = 1;

    println!("gradient {:?}, gradient norm {}", gradient, gradient_norm(&gradient));
    //check if its the minimum
    while gradient_norm(&gradient) > tolerance{
        println!("{}: Gradient vec: [{}, {}, {}]", iter, gradient[0], gradient[1], gradient[2]);
        
        //move in that direction a set step size 
        move_from_gradient(p, &gradient, step_size);

        println!("Moved to: ({}, {}, {})", p.x, p.y, p.z);
        //calculate a direction to go in
        gradient = calc_gradient(p.clone(), r);

        iter += 1;
    } 

    p.clone()
}

fn calc_gradient(p: Point, r: f64) -> Vec<f64> {
    let h = 1e-5; // Small step size for finite differences

    // Fault function value at the current point
    let f_value = fault_function(p.clone(), r);

    // Gradient components
    let mut gradient = vec![0.0; 3];

    // Partial derivative with respect to x
    let mut perturbed_p = p.clone();
    perturbed_p.x += h;
    gradient[0] = (fault_function(perturbed_p, r) - f_value) / h;

    // Partial derivative with respect to y
    perturbed_p = p.clone();
    perturbed_p.y += h;
    gradient[1] = (fault_function(perturbed_p, r) - f_value) / h;

    // Partial derivative with respect to z
    perturbed_p = p.clone();
    perturbed_p.z += h;
    gradient[2] = (fault_function(perturbed_p, r) - f_value) / h;

    gradient
}

fn gradient_norm(g: &Vec<f64>) -> f64{
    (g[0].powi(2) + g[1].powi(2) + g[2].powi(2)).sqrt()
}

fn move_from_gradient(p: &mut Point, g: &Vec<f64>, s: f64){
    p.x -= s*g[0];
    p.y -= s*g[1];
    p.z -= s*g[2];
}