#[derive(Clone, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64, 
    pub z: f64
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
    }


    //this is a minimization problem so we take the opposite value of our volume
    pub fn F(&self) -> f64{
        -1.0*self.x*self.y*self.z
    }
    
    //G1 evaluates the requrement for the side area sum to be equal to 1
    fn G1(&self) -> f64{
        let out = (self.x * self.y + self.x * self.z + self.z * self.y)*2.0 - 1.0;
        if out.is_infinite(){panic!("g1 panicked");};

        out
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
        let max_H_squared = f64::max(Self::H1(self), f64::max(Self::H2(self), Self::H3(self))).powi(2);

        sum_G_squared+max_H_squared
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
        let grad_G1_x = (2.0 / r) * self.G1() * partial_G1_x(self.x, self.y, self.z);
        let grad_G1_y = (2.0 / r) * self.G1() * partial_G1_y(self.x, self.y, self.z);
        let grad_G1_z = (2.0 / r) * self.G1() * partial_G1_z(self.x, self.y, self.z);

        // Gradients of H^2 (use the maximum constraint, as B includes only the max H term)
        let max_H = f64::max(self.H1(), f64::max(self.H2(), self.H3()));
        let grad_H_x = if max_H == self.H1() {
            (2.0 / r) * max_H * partial_H1(self.x)
        } else {
            0.0
        };
        let grad_H_y = if max_H == self.H2() {
            (2.0 / r) * max_H * partial_H2(self.y)
        } else {
            0.0
        };
        let grad_H_z = if max_H == self.H3() {
            (2.0 / r) * max_H * partial_H3(self.z)
        } else {
            0.0
        };

        // Combine gradients
        let grad_x = grad_G1_x + grad_H_x;
        let grad_y = grad_G1_y + grad_H_y;
        let grad_z = grad_G1_z + grad_H_z;

        // Return the gradient as a Point
        Point { x: grad_x, y: grad_y, z: grad_z }
    }

    pub fn move_towards_gradient(&self, gradient: Self, gama: f64) -> Self{
        Self {
            x: self.x + gradient.x * gama,
            y: self.y + gradient.y * gama,
            z: self.z + gradient.z * gama
        }
    }
    
}

pub fn calc_gradient_norm(gradient_vec: Point) -> f64{
    (gradient_vec.x.powf(2.0)+gradient_vec.y.powf(2.0)+gradient_vec.z.powf(2.0)).sqrt()
}