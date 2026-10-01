#include "product/media_acquire/acquire.h"

#include <stdexcept>

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::invalid_argument(
        "media acquisition is unavailable in the ninfer-windows-serve text-only build");
}

} // namespace ninfer::product::media_acquire
