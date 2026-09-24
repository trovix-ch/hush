# Included after every project() in the llama.cpp build. Its Vulkan shader generator is a
# nested ExternalProject whose default prefix adds ~45 characters to already deep cargo
# paths, and MSVC fails with C1083 past MAX_PATH even with Windows long paths enabled.
set_property(DIRECTORY PROPERTY EP_PREFIX "${CMAKE_BINARY_DIR}/e")
