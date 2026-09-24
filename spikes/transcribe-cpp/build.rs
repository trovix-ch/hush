// transcribe-cpp-sys 0.2.3 emits `vulkan-1.lib` as a bare name without a search path,
// so the final link fails with LNK1181 unless the SDK's Lib directory is added here.
fn main() {
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let lib = std::path::Path::new(&sdk).join("Lib");
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
}
