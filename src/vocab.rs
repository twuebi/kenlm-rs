use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct BidiMapping {
    reverse: Vec<String>,
    s2val: HashMap<String, u32>,
}

impl BidiMapping {
    pub fn insert_or_get(&mut self, key: String) -> u32 {
        *self.s2val.entry(key).or_insert_with_key(|key| {
            let idx = self.reverse.len() as u32;
            self.reverse.push(key.clone());
            idx
        })
    }

    pub fn get_index(&self, key: &str) -> Option<u32> {
        self.s2val.get(key).cloned()
    }

    pub fn get_value(&mut self, index: u32) -> Option<&str> {
        self.reverse.get(index as usize).map(|s| s.as_str())
    }

    pub fn values(&mut self) -> &[String] {
        &self.reverse
    }

    pub fn len(&self) -> usize {
        return self.reverse.len()
    }
}
