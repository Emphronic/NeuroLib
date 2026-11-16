#![allow(dead_code)]

use std::path::PathBuf;

/// Unique temporary test directory; is deleted at `Drop`.
pub struct TmpDir(pub PathBuf);

impl TmpDir {
    pub fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("syn-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TmpDir(p)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
