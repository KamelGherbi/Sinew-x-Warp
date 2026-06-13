fn main() {
    println!("cargo:rerun-if-env-changed=SINEW_REMOTE_RELAY_URL");
    tauri_build::build()
}
