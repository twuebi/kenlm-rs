use std::collections::HashMap;

pub trait BidirectionalMapping<INDEX, VALUE> {
    fn insert_or_get_index(&mut self, value: VALUE) -> INDEX;
    fn get_index(&self, value: &VALUE) -> Option<&INDEX>;
    fn get_value(&self, index: INDEX) -> Option<&VALUE>;
    fn len(&self) -> usize;
}
#[derive(Debug)]
pub enum Mappings {
    HashVecMap(HashMapVecMap),
    NoOp,
}

impl Default for Mappings {
    fn default() -> Self {
        Self::HashVecMap(Default::default())
    }
}

pub struct NoOPMapping;
impl<INDEX, VALUE> BidirectionalMapping<INDEX, VALUE> for NoOPMapping
where
    INDEX: Default,
    VALUE: Default,
{
    fn insert_or_get_index(&mut self, _: VALUE) -> INDEX {
        Default::default()
    }

    fn get_index(&self, _: &VALUE) -> Option<&INDEX> {
        Default::default()
    }

    fn get_value(&self, _: INDEX) -> Option<&VALUE> {
        Default::default()
    }

    fn len(&self) -> usize {
        Default::default()
    }
}

impl BidirectionalMapping<u32, String> for Mappings {
    fn insert_or_get_index(&mut self, value: String) -> u32 {
        match self {
            Mappings::HashVecMap(hm) => hm.insert_or_get_index(value),
            _ => Default::default(),
        }
    }

    fn get_index(&self, value: &String) -> Option<&u32> {
        match self {
            Mappings::HashVecMap(hm) => hm.get_index(value),
            _ => Default::default(),
        }
    }

    fn get_value(&self, index: u32) -> Option<&String> {
        match self {
            Mappings::HashVecMap(hm) => hm.get_value(index),
            _ => Default::default(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Mappings::HashVecMap(hm) => hm.len(),
            _ => Default::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct HashMapVecMap {
    reverse: Vec<String>,
    s2val: HashMap<String, u32>,
}

impl BidirectionalMapping<u32, String> for HashMapVecMap {
    fn insert_or_get_index(&mut self, value: String) -> u32 {
        *self.s2val.entry(value).or_insert_with_key(|value| {
            let idx = self.reverse.len() as u32;
            self.reverse.push(value.clone());
            idx
        })
    }
    fn get_index(&self, key: &String) -> Option<&u32> {
        self.s2val.get(key)
    }
    fn get_value(&self, index: u32) -> Option<&String> {
        self.reverse.get(index as usize)
    }
    fn len(&self) -> usize {
        self.reverse.len()
    }
}
