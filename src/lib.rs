pub extern crate nalgebra;

use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, OPoint, OVector, RealField, U1, U2, U3,
};

pub type BoundingSpace1<T> = BoundingSpaceN<T, U1>;
pub type BoundingSpace2<T> = BoundingSpaceN<T, U2>;
pub type BoundingSpace3<T> = BoundingSpaceN<T, U3>;

pub type BoundingRange<T> = BoundingSpace1<T>;
pub type BoundingSquare<T> = BoundingSpace2<T>;
pub type BoundingBox<T> = BoundingSpace3<T>;

#[derive(Debug, Clone)]
pub struct BoundingSpaceN<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub lower: OPoint<T, D>,
    pub upper: OPoint<T, D>,
}

impl<T, D> BoundingSpaceN<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn new(lower: OPoint<T, D>, upper: OPoint<T, D>) -> Self {
        Self { lower, upper }
    }

    pub fn from_point(point: OPoint<T, D>) -> Self {
        Self::new(point.to_owned(), point)
    }

    pub fn from_value(value: T) -> Self {
        Self {
            lower: OVector::repeat(value.to_owned()).into(),
            upper: OVector::repeat(value).into(),
        }
    }

    pub fn from_values(lower: T, upper: T) -> Self {
        Self {
            lower: OVector::repeat(lower).into(),
            upper: OVector::repeat(upper).into(),
        }
    }

    pub fn diagonal(&self) -> OVector<T, D> {
        &self.upper - &self.lower
    }

    pub fn contains(&self, point: &OPoint<T, D>) -> bool {
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

    pub fn expand_lower(&mut self, point: &OPoint<T, D>) {
        self.lower.coords.zip_apply(&point.coords, |l, p| {
            *l = p.min(l.to_owned());
        });
    }

    pub fn expand_upper(&mut self, point: &OPoint<T, D>) {
        self.upper.coords.zip_apply(&point.coords, |u, p| {
            *u = p.max(u.to_owned());
        });
    }

    pub fn expand(&mut self, point: &OPoint<T, D>) {
        self.expand_lower(point);
        self.expand_upper(point);
    }
}

impl<T, D> Default for BoundingSpaceN<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn default() -> Self {
        Self {
            lower: OVector::repeat(T::zero()).into(),
            upper: OVector::repeat(T::zero()).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Point1, U1};

    use super::*;

    #[test]
    fn default_initialized_with_zeros() {
        let bs = BoundingSpaceN::<f64, U1>::default();

        for v in bs.lower.iter().chain(bs.upper.iter()) {
            assert_relative_eq!(*v, 0.0);
        }
    }

    #[test]
    fn first_expand_nan() {
        let mut bound = BoundingSpaceN::<f64, U1>::from_value(f64::NAN);
        let point = Point1::new(0.0);

        bound.expand(&point);

        assert_relative_eq!(bound.lower.x, point.x);
        assert_relative_eq!(bound.upper.x, point.x);
    }

    #[test]
    fn expand_1d() {
        let value = 0.0_f64;
        let mut bound = BoundingSpaceN::<f64, U1>::from_value(value);

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
