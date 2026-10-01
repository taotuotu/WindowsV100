#include "media/decode/decode.h"

namespace ninfer::media::decode {
namespace {

[[noreturn]] void unsupported() {
    throw Error(ErrorKind::InvalidInput,
                "media decoding is unavailable in the text-only Windows server build");
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t>, const Policy&) { unsupported(); }

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unsupported();
}

Image decode_image(std::span<const std::uint8_t>, const Policy&) { unsupported(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unsupported();
}

} // namespace ninfer::media::decode
