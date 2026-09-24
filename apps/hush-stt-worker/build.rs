// transcribe-cpp-sys 0.2.3 links `vulkan-1.lib` by bare name without a search path, so
// the Windows link fails with LNK1181 unless the SDK's Lib directory is added here.
fn main() {
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    let windows = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows");
    match std::env::var_os("VULKAN_SDK") {
        Some(sdk) => {
            let lib = std::path::Path::new(&sdk).join("Lib");
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
        None if windows => panic!(
            "hush-stt-worker: VULKAN_SDK is not set. Install the Vulkan SDK (e.g. \
             `winget install KhronosGroup.VulkanSDK`) and open a fresh shell, or set \
             VULKAN_SDK to its install directory, e.g. C:\\VulkanSDK\\1.4.357.0."
        ),
        None => {}
    }
}
