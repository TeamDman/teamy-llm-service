use std::process::Command;

fn git(args: &[&str]) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    Command::new("git")
        .current_dir(root)
        .arg("-c")
        .arg(format!("safe.directory={}", root.display()))
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!("cargo:rerun-if-changed=src");
    println!(
        "cargo:rustc-env=GIT_REVISION={}",
        git(&["rev-parse", "HEAD"])
    );
    println!(
        "cargo:rustc-env=GIT_BRANCH={}",
        git(&["branch", "--show-current"])
    );
    let status = git(&["status", "--porcelain"]);
    println!(
        "cargo:rustc-env=GIT_WORKTREE_STATUS={}",
        match status.as_str() {
            "" => "clean",
            "unknown" => "unknown",
            _ => "dirty",
        }
    );
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    let timestamp = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_secs())
        });
    println!("cargo:rustc-env=BUILD_TIMESTAMP_UNIX={timestamp}");
}
