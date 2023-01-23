pub extern crate nalgebra;

use nalgebra::{SVector, RealField, Point};

pub type BoundingSpace1<T> = BoundingSpaceN<T, 1>;
pub type BoundingSpace2<T> = BoundingSpaceN<T, 2>;
pub type BoundingSpace3<T> = BoundingSpaceN<T, 3>;

pub type BoundingRange<T> = BoundingSpace1<T>;
pub type BoundingSquare<T> = BoundingSpace2<T>;
pub type BoundingBox<T> = BoundingSpace3<T>;

#[derive(Debug, Clone, Copy)]
pub struct BoundingSpaceN<T: RealField, const D: usize> {
    pub lower: Point<T, D>,
    pub upper: Point<T, D>,
}

impl<T: RealField, const D: usize> BoundingSpaceN<T, D> {
    pub fn new(lower: Point<T, D>, upper: Point<T, D>) -> Self {
        Self { lower, upper }
    }

    pub fn from_point(point: Point<T, D>) -> Self {
        Self::new(point.to_owned(), point)
    }

    pub fn from_value(value: T) -> Self {
        let coords =  SVector::<T, D>::repeat(value);
        Self {
            lower: Point { coords: coords.to_owned() },
            upper: Point { coords }
        }
    }

    pub fn from_values(lower: T, upper: T) -> Self {
        Self {
            lower: Point { coords: SVector::repeat(lower) },
            upper: Point { coords: SVector::repeat(upper) },
        }
    }

    pub fn diagonal(&self) -> SVector<T, D> {
        &self.upper - &self.lower
    }

    pub fn contains(&self, point: &Point<T, D>) -> bool {
        for (l, p) in self.lower.coords.iter().zip(&point.coords) {
            if l > p {
                return false;
            }
        }

        for (u, p) in self.upper.coords.iter().zip(&point.coords) {
            if u < p {
                return false;
            }
        }

        true
    }

    pub fn expand_lower(&mut self, point: &Point<T, D>) {
        self.lower.coords.zip_apply(&point.coords, |l, p| {
            *l = p.min(l.to_owned());
        });
    }

    pub fn expand_upper(&mut self, point: &Point<T, D>) {
        self.upper.coords.zip_apply(&point.coords, |u, p| {
            *u = p.max(u.to_owned());
        });
    }

    pub fn expand(&mut self, point: &Point<T, D>) {
        self.expand_lower(point);
        self.expand_upper(point);
    }
}

impl<T: RealField, const D: usize> Default for BoundingSpaceN<T, D>
{
    fn default() -> Self {
        Self {
            lower: Point::origin(),
            upper: Point::origin(),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Point1;

    use super::*;

    #[test]
    fn default_initialized_with_zeros() {
        let bs = BoundingSpaceN::<f64, 1>::default();

        for v in bs.lower.iter().chain(bs.upper.iter()) {
            assert_relative_eq!(*v, 0.0);
        }
    }

    #[test]
    fn first_expand_nan() {
        let mut bound = BoundingSpaceN::<f64, 1>::from_value(f64::NAN);
        let point = Point1::new(0.0);

        bound.expand(&point);

        assert_relative_eq!(bound.lower.x, point.x);
        assert_relative_eq!(bound.upper.x, point.x);
    }

    #[test]
    fn expand_1d() {
        let value = 0.0_f64;
        let mut bound = BoundingSpaceN::<f64, 1>::from_value(value);

        let p1 = Point1::new(1.0);
        bound.expand(&p1);

        assert_relative_eq!(bound.lower.x, value);
        assert_relative_eq!(bound.upper.x, p1.x);

        let p2 = Point1::new(-1.0);
        bound.expand(&p2);

        assert_relative_eq!(bound.lower.x, p2.x);
        assert_relative_eq!(bound.upper.x, p1.x);
    }
}
