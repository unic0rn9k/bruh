use tch::Tensor;
use std::collections::HashMap;
//use serde::{Serialize, Deserialize};

#[derive(Default, Debug)]
pub struct SpatialHashGrid{
    nodes: HashMap<Box<[i64]>, Vec<(Tensor, usize)>>,
    node_size: f64,
}

impl SpatialHashGrid{
    pub fn insert(&mut self, position: Tensor, value: usize){
        assert!(position.size().len() == 2);
        assert!(position.size()[0] == 1);

        let mut index = vec![0; position.size()[1] as usize].into_boxed_slice();
        for i in 0..position.size().len(){
            index[i] = (position.double_value(&[0, i as i64]) / self.node_size).floor() as i64;
        }

        if self.nodes.contains_key(&index){
            self.nodes.get_mut(&index).unwrap().push((position, value));
        }else{
            self.nodes.insert(index, vec![(position, value)]);
        }
    }

    pub fn neighbors(&self, query: &Tensor) -> &[(Tensor, usize)]{
        let mut index = vec![0; query.size()[1] as usize].into_boxed_slice();
        for i in 0..query.size().len(){
            index[i] = (query.double_value(&[0, i as i64]) / self.node_size).floor() as i64;
        }

        if self.nodes.contains_key(&index){
            self.nodes.get(&index).unwrap()
        }else{
            &[]
        }
    }
}

#[test]
fn test_spatial_hash_grid(){
    let mut grid = SpatialHashGrid::default();
    grid.node_size = 2.;

    grid.insert(Tensor::from_slice(&[1., 2., 4.]).reshape([1,3]), 1);
    grid.insert(Tensor::from_slice(&[0., 0., 0.]).reshape([1,3]), 2);
    grid.insert(Tensor::from_slice(&[1., 0., 0.]).reshape([1,3]), 3);

    let check = |x, y: &[usize]| {
        let neighbors = grid.neighbors(&Tensor::from_slice(x).reshape([1,3]));
        assert!(
            neighbors.iter().zip(y.iter()).all(|(x,y)|x.1==*y),
            "{:#?}", grid
        );
    };

    println!("{:?}", Tensor::from_slice(&[1., 2., 4.]).size());

    check(&[1., 2., 3.], &[1]);
    check(&[0., 0., 0.], &[2, 3]);
}
