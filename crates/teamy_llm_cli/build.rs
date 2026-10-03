use std::process::Command;

#[expect(
    clippy::disallowed_macros,
    reason = "Build-script stdout is Cargo's instruction protocol, not runtime logging"
)]
fn cargo_instruction(instruction: std::fmt::Arguments<'_>) {
    println!("cargo:{instruction}");
}

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
    cargo_instruction(format_args!("rerun-if-changed=../../.git/HEAD"));
    cargo_instruction(format_args!("rerun-if-changed=../../.git/index"));
    cargo_instruction(format_args!("rerun-if-changed=src"));
    cargo_instruction(format_args!(
        "rustc-env=GIT_REVISION={}",
        git(&["rev-parse", "HEAD"])
    ));
    cargo_instruction(format_args!(
        "rustc-env=GIT_BRANCH={}",
        git(&["branch", "--show-current"])
    ));
    let status = git(&["status", "--porcelain"]);
    cargo_instruction(format_args!(
        "rustc-env=GIT_WORKTREE_STATUS={}",
        match status.as_str() {
            "" => "clean",
            "unknown" => "unknown",
            _ => "dirty",
        }
    ));
    cargo_instruction(format_args!("rerun-if-env-changed=SOURCE_DATE_EPOCH"));
    let timestamp = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_secs())
        });
    cargo_instruction(format_args!("rustc-env=BUILD_TIMESTAMP_UNIX={timestamp}"));
}
