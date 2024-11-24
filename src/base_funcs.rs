#[derive(Clone, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64, 
    pub z: f64
}

impl Point{
    //this is a minimization problem so we take the opposite value of our volume
    pub fn F(&self) -> f64{
        -1.0*self.x*self.y*self.z
    }
    
    //G1 evaluates the requrement for the side area sum to be equal to 1
    pub fn G1(&self) -> f64{
        fn calc_s(x:f64, y:f64) -> f64{
            x*y
        }
    
        (calc_s(self.x, self.y)+calc_s(self.x, self.z)+calc_s(self.z, self.y))*2.0 - 1.0
    }
    
    // x > 0 constraint
    pub fn H1(&self) -> f64 {
        f64::max(0.0, -1.0 * self.x)
    }

    // y > 0 constraint
    pub fn H2(&self) -> f64 {
        f64::max(0.0, -1.0 * self.y)
    }

    // z > 0 constraint
    pub fn H3(&self) -> f64 {
        f64::max(0.0, -1.0 * self.z)
}
}
