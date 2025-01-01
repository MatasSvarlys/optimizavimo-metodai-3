#[derive(Clone, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64, 
    pub z: f64
}

pub fn check_value(out: f64) -> Result<(), &'static str> {
    if out.is_infinite() {
        // eprintln!("g1 got out of bounds");
        return Err("Infinite value detected");
    }
    Ok(())
}
impl Point{
    pub fn print(&self){
        println!("Point: {}, {}, {}", self.x, self.y, self.z);
    }
    pub fn print_constraints(&self){
        println!("G={}; Hi=(1={}, 2={}, 3={})", self.G1(), self.H1(), self.H2(), self.H3());
    }
    pub fn print_all(&self){
        self.print();
        self.print_constraints();
        println!("F: {}", self.F());
        println!()
    }


    //this is a minimization problem so we take the opposite value of our volume
    pub fn F(&self) -> f64{
        -1.0*self.x*self.y*self.z
    }
    
    //G1 evaluates the requrement for the side area sum to be equal to 1
    pub fn G1(&self) -> f64{
        (self.x * self.y + self.x * self.z + self.z * self.y)*2.0 - 1.0
    }
    
    // x > 0 constraint
    fn H1(&self) -> f64 {
        f64::max(0.0, -1.0 * self.x)
    }

    // y > 0 constraint
    fn H2(&self) -> f64 {
        f64::max(0.0, -1.0 * self.y)
    }

    // z > 0 constraint
    fn H3(&self) -> f64 {
        f64::max(0.0, -1.0 * self.z)
    }

    pub fn B(&self) -> f64{
        let sum_G_squared = Self::G1(self).powi(2);    
        let sum_H_squared = Self::H1(self).powi(2) + Self::H2(self).powi(2) + Self::H3(self).powi(2);

        sum_G_squared+sum_H_squared
    }

    pub fn gradient_B(&self, r: f64) -> Point {
        // Helper function to compute the partial derivative of G1 with respect to a variable
        fn partial_G1_x(x: f64, y: f64, z: f64) -> f64 {
            2.0 * y + 2.0 * z
        }
        fn partial_G1_y(x: f64, y: f64, z: f64) -> f64 {
            2.0 * x + 2.0 * z
        }
        fn partial_G1_z(x: f64, y: f64, z: f64) -> f64 {
            2.0 * y + 2.0 * x
        }

        // Partial derivatives of H1, H2, H3
        //If Hi < 0 -> H' = (-x)' = -1 else Hi = 0 -> (0)' = 0 
        fn partial_H1(x: f64) -> f64 {
            if x < 0.0 { -1.0 } else { 0.0 }
        }
        fn partial_H2(y: f64) -> f64 {
            if y < 0.0 { -1.0 } else { 0.0 }
        }
        fn partial_H3(z: f64) -> f64 {
            if z < 0.0 { -1.0 } else { 0.0 }
        }

        // Gradients of G1^2
        let grad_G1_x = (2.0 / r) * self.G1() * partial_G1_x(self.x, self.y, self.z); //G1^2'=2*G1*G1'
        let grad_G1_y = (2.0 / r) * self.G1() * partial_G1_y(self.x, self.y, self.z);
        let grad_G1_z = (2.0 / r) * self.G1() * partial_G1_z(self.x, self.y, self.z);

        let grad_H_x = (2.0 / r) * self.H1() * partial_H1(self.x); 
        let grad_H_y = (2.0 / r) * self.H2() * partial_H2(self.y); 
        let grad_H_z = (2.0 / r) * self.H3() * partial_H3(self.z); 
    
        // Combine gradients
        let grad_x = grad_G1_x + grad_H_x;
        let grad_y = grad_G1_y + grad_H_y;
        let grad_z = grad_G1_z + grad_H_z;

        // Return the gradient as a Point
        Point { x: grad_x, y: grad_y, z: grad_z }
    }

    pub fn move_towards_gradient(&self, gradient: Self, gama: f64) -> Self{
        Self {
            x: self.x - gradient.x * gama,
            y: self.y - gradient.y * gama,
            z: self.z - gradient.z * gama
        }
    }
    pub fn gradient_F(&self) -> Point{
        Point {
            x: -self.y * self.z,
            y: -self.x * self.z,
            z: -self.x * self.y,
        }
    }

    pub fn gradient_full(&self, r: f64) -> Point{
        let grad_F = self.gradient_F(); // Gradient of the base function
        let grad_B = self.gradient_B(r); // Gradient of the penalty term
    
        Point {
            x: grad_F.x + grad_B.x,
            y: grad_F.y + grad_B.y,
            z: grad_F.z + grad_B.z,
        }
    }
}

pub fn calc_gradient_norm(gradient: Point) -> f64{
    (gradient.x.powi(2) + gradient.y.powi(2) + gradient.z.powi(2)).sqrt()
}