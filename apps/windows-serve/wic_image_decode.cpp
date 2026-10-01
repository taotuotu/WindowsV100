// Windows/V100 implementation by taotuotu, 2026; see NOTICE.
#include "media/decode/decode.h"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::media::decode {
namespace {

using Microsoft::WRL::ComPtr;

[[noreturn]] void throw_invalid_media(std::string message) {
    throw Error(ErrorKind::InvalidInput, std::move(message));
}

void validate_input(std::span<const std::uint8_t> bytes, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    if (bytes.empty()) { throw_invalid_media("image bytes are empty"); }
    if (bytes.size() > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
    }
    if (bytes.size() > std::numeric_limits<DWORD>::max()) {
        throw Error(ErrorKind::BudgetExceeded, "image bytes exceed Windows decoder buffer limit");
    }
}

int exif_orientation(std::span<const std::uint8_t> bytes) {
    constexpr std::size_t limit = 1ULL << 20;
    bytes                       = bytes.first(std::min(limit, bytes.size()));
    if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) { return 1; }
    const auto be16 = [&](std::size_t offset) {
        return static_cast<std::uint16_t>((bytes[offset] << 8U) | bytes[offset + 1]);
    };

    std::size_t marker = 2;
    while (marker + 4 <= bytes.size()) {
        if (bytes[marker] != 0xff) { break; }
        const std::uint8_t kind = bytes[marker + 1];
        if (kind == 0xda || kind == 0xd9) { break; }
        const std::uint16_t length = be16(marker + 2);
        if (length < 2 || marker + 2 + length > bytes.size()) { break; }
        const std::size_t payload      = marker + 4;
        const std::size_t payload_size = length - 2;
        if (kind == 0xe1 && payload_size >= 14 &&
            std::memcmp(bytes.data() + payload, "Exif\0\0", 6) == 0) {
            const std::size_t tiff = payload + 6;
            const bool little      = bytes[tiff] == 'I' && bytes[tiff + 1] == 'I';
            const bool big         = bytes[tiff] == 'M' && bytes[tiff + 1] == 'M';
            if (!little && !big) { return 1; }
            const auto u16 = [&](std::size_t offset) -> std::uint16_t {
                if (offset + 2 > bytes.size()) { return 0; }
                return little ? static_cast<std::uint16_t>(bytes[offset] | bytes[offset + 1] << 8U)
                              : static_cast<std::uint16_t>(bytes[offset] << 8U | bytes[offset + 1]);
            };
            const auto u32 = [&](std::size_t offset) -> std::uint32_t {
                if (offset + 4 > bytes.size()) { return 0; }
                if (little) {
                    return static_cast<std::uint32_t>(bytes[offset]) |
                           static_cast<std::uint32_t>(bytes[offset + 1]) << 8U |
                           static_cast<std::uint32_t>(bytes[offset + 2]) << 16U |
                           static_cast<std::uint32_t>(bytes[offset + 3]) << 24U;
                }
                return static_cast<std::uint32_t>(bytes[offset]) << 24U |
                       static_cast<std::uint32_t>(bytes[offset + 1]) << 16U |
                       static_cast<std::uint32_t>(bytes[offset + 2]) << 8U |
                       static_cast<std::uint32_t>(bytes[offset + 3]);
            };
            if (u16(tiff + 2) != 42) { return 1; }
            const std::size_t ifd = tiff + u32(tiff + 4);
            if (ifd + 2 > bytes.size()) { return 1; }
            const std::uint16_t count = u16(ifd);
            for (std::uint16_t i = 0; i < count; ++i) {
                const std::size_t entry = ifd + 2 + static_cast<std::size_t>(i) * 12;
                if (entry + 12 > bytes.size()) { return 1; }
                if (u16(entry) == 0x0112 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
                    const int orientation = u16(entry + 8);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
            return 1;
        }
        marker += 2 + length;
    }
    return 1;
}

class ComApartment final {
public:
    ComApartment() {
        const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (result == RPC_E_CHANGED_MODE) { return; }
        if (FAILED(result)) { throw std::runtime_error("failed to initialize Windows Imaging Component"); }
        initialized_ = true;
    }

    ~ComApartment() {
        if (initialized_) { CoUninitialize(); }
    }

    ComApartment(const ComApartment&)            = delete;
    ComApartment& operator=(const ComApartment&) = delete;

private:
    bool initialized_ = false;
};

struct Dimensions {
    int width  = 0;
    int height = 0;
};

Dimensions checked_dimensions(UINT width, UINT height, const Policy& policy) {
    if (width == 0 || height == 0) { throw_invalid_media("decoded image dimensions are invalid"); }
    if (width > static_cast<UINT>(std::numeric_limits<int>::max()) ||
        height > static_cast<UINT>(std::numeric_limits<int>::max())) {
        throw Error(ErrorKind::BudgetExceeded, "decoded image dimensions exceed processor limit");
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels > policy.max_decoded_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    if (pixels > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 3)) {
        throw Error(ErrorKind::BudgetExceeded, "decoded image storage exceeds addressable memory");
    }
    return Dimensions{static_cast<int>(width), static_cast<int>(height)};
}

Image orient_image(Image image, int orientation) {
    if (orientation == 1) { return image; }
    const int width  = image.width;
    const int height = image.height;
    const bool swap  = orientation >= 5;
    Image rotated;
    rotated.width  = swap ? height : width;
    rotated.height = swap ? width : height;
    rotated.rgb.resize(image.rgb.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int rx = 0;
            int ry = 0;
            if (orientation == 2) {
                rx = width - 1 - x;
                ry = y;
            } else if (orientation == 3) {
                rx = width - 1 - x;
                ry = height - 1 - y;
            } else if (orientation == 4) {
                rx = x;
                ry = height - 1 - y;
            } else if (orientation == 5) {
                rx = y;
                ry = x;
            } else if (orientation == 6) {
                rx = height - 1 - y;
                ry = x;
            } else if (orientation == 7) {
                rx = height - 1 - y;
                ry = width - 1 - x;
            } else { // orientation 8
                rx = y;
                ry = width - 1 - x;
            }
            const std::size_t source = (static_cast<std::size_t>(y) * width + x) * 3;
            const std::size_t target =
                (static_cast<std::size_t>(ry) * rotated.width + rx) * 3;
            std::copy_n(image.rgb.data() + source, 3, rotated.rgb.data() + target);
        }
    }
    return rotated;
}

class WicImageDecoder final {
public:
    WicImageDecoder(std::span<const std::uint8_t> bytes, const Policy& policy)
        : bytes_(bytes), policy_(policy), orientation_(exif_orientation(bytes)) {
        HRESULT result = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                          IID_PPV_ARGS(factory_.GetAddressOf()));
        if (FAILED(result)) {
            throw std::runtime_error("Windows Imaging Component factory is unavailable");
        }
        result = factory_->CreateStream(stream_.GetAddressOf());
        if (FAILED(result)) { throw std::runtime_error("failed to create Windows image stream"); }
        result = stream_->InitializeFromMemory(
            reinterpret_cast<BYTE*>(const_cast<std::uint8_t*>(bytes_.data())),
            static_cast<DWORD>(bytes_.size()));
        if (FAILED(result)) { throw_invalid_media("failed to open image bytes"); }
        result = factory_->CreateDecoderFromStream(stream_.Get(), nullptr,
                                                    WICDecodeMetadataCacheOnDemand,
                                                    decoder_.GetAddressOf());
        if (FAILED(result)) {
            throw_invalid_media("image is not a decodable JPEG, PNG, or BMP file");
        }

        GUID container{};
        result = decoder_->GetContainerFormat(&container);
        if (FAILED(result) ||
            (!IsEqualGUID(container, GUID_ContainerFormatJpeg) &&
             !IsEqualGUID(container, GUID_ContainerFormatPng) &&
             !IsEqualGUID(container, GUID_ContainerFormatBmp))) {
            throw_invalid_media("Windows image input supports JPEG, PNG, and BMP only");
        }

        UINT frame_count = 0;
        result           = decoder_->GetFrameCount(&frame_count);
        if (FAILED(result) || frame_count == 0) { throw_invalid_media("image contains no frame"); }
        result = decoder_->GetFrame(0, frame_.GetAddressOf());
        if (FAILED(result)) { throw_invalid_media("failed to decode the first image frame"); }

        UINT width  = 0;
        UINT height = 0;
        result      = frame_->GetSize(&width, &height);
        if (FAILED(result)) { throw_invalid_media("failed to inspect image dimensions"); }
        dimensions_ = checked_dimensions(width, height, policy_);
    }

    [[nodiscard]] ImageInfo info() const noexcept {
        const bool swap = orientation_ >= 5;
        return ImageInfo{.width = swap ? dimensions_.height : dimensions_.width,
                         .height = swap ? dimensions_.width : dimensions_.height};
    }

    void validate_decoded_frame() {
        WICPixelFormatGUID pixel_format{};
        HRESULT result = frame_->GetPixelFormat(&pixel_format);
        if (FAILED(result)) { throw_invalid_media("failed to inspect image pixel format"); }

        ComPtr<IWICComponentInfo> component_info;
        result = factory_->CreateComponentInfo(pixel_format, component_info.GetAddressOf());
        if (FAILED(result)) { throw_invalid_media("image pixel format is not supported"); }
        ComPtr<IWICPixelFormatInfo> format_info;
        result = component_info.As(&format_info);
        if (FAILED(result)) { throw_invalid_media("image pixel format is not supported"); }
        UINT bits_per_pixel = 0;
        result               = format_info->GetBitsPerPixel(&bits_per_pixel);
        if (FAILED(result) || bits_per_pixel == 0) {
            throw_invalid_media("image pixel format has invalid dimensions");
        }

        const std::uint64_t row_bits =
            static_cast<std::uint64_t>(dimensions_.width) * bits_per_pixel;
        const std::uint64_t stride = (row_bits + 7U) / 8U;
        if (stride > std::numeric_limits<UINT>::max() ||
            static_cast<std::uint64_t>(dimensions_.height) >
                std::numeric_limits<std::uint64_t>::max() / stride) {
            throw Error(ErrorKind::BudgetExceeded, "decoded image storage exceeds WIC buffer limit");
        }
        const std::uint64_t buffer_size = stride * static_cast<std::uint64_t>(dimensions_.height);
        if (buffer_size > std::numeric_limits<UINT>::max()) {
            throw Error(ErrorKind::BudgetExceeded, "decoded image storage exceeds WIC buffer limit");
        }
        std::vector<std::uint8_t> decoded(static_cast<std::size_t>(buffer_size));
        result = frame_->CopyPixels(nullptr, static_cast<UINT>(stride),
                                    static_cast<UINT>(buffer_size), decoded.data());
        if (FAILED(result)) { throw_invalid_media("failed to decode image pixels"); }
    }

    [[nodiscard]] Image decode_rgb() {
        ComPtr<IWICFormatConverter> converter;
        HRESULT result = factory_->CreateFormatConverter(converter.GetAddressOf());
        if (FAILED(result)) { throw std::runtime_error("failed to create Windows RGB converter"); }
        WICPixelFormatGUID source_format{};
        result = frame_->GetPixelFormat(&source_format);
        if (FAILED(result)) { throw_invalid_media("failed to inspect image pixel format"); }
        BOOL can_convert = FALSE;
        result = converter->CanConvert(source_format, GUID_WICPixelFormat24bppRGB, &can_convert);
        if (FAILED(result) || can_convert == FALSE) {
            throw_invalid_media("image pixel format cannot be converted to RGB24");
        }
        result = converter->Initialize(frame_.Get(), GUID_WICPixelFormat24bppRGB,
                                       WICBitmapDitherTypeNone, nullptr, 0.0,
                                       WICBitmapPaletteTypeCustom);
        if (FAILED(result)) { throw_invalid_media("failed to convert image to RGB24"); }

        const std::uint64_t stride = static_cast<std::uint64_t>(dimensions_.width) * 3U;
        const std::uint64_t buffer_size = stride * static_cast<std::uint64_t>(dimensions_.height);
        if (stride > std::numeric_limits<UINT>::max() ||
            buffer_size > std::numeric_limits<UINT>::max() ||
            buffer_size > std::numeric_limits<std::size_t>::max()) {
            throw Error(ErrorKind::BudgetExceeded, "decoded RGB image exceeds WIC buffer limit");
        }

        Image image;
        image.width  = dimensions_.width;
        image.height = dimensions_.height;
        image.rgb.resize(static_cast<std::size_t>(buffer_size));
        result = converter->CopyPixels(nullptr, static_cast<UINT>(stride),
                                       static_cast<UINT>(buffer_size), image.rgb.data());
        if (FAILED(result)) { throw_invalid_media("failed to decode image pixels as RGB24"); }
        return orient_image(std::move(image), orientation_);
    }

private:
    ComApartment apartment_;
    ComPtr<IWICImagingFactory> factory_;
    ComPtr<IWICStream> stream_;
    ComPtr<IWICBitmapDecoder> decoder_;
    ComPtr<IWICBitmapFrameDecode> frame_;
    std::span<const std::uint8_t> bytes_;
    const Policy& policy_;
    Dimensions dimensions_;
    int orientation_ = 1;
};

[[noreturn]] void unsupported_video() {
    throw Error(ErrorKind::InvalidInput,
                "video decoding is unavailable in the Windows image-only server build");
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    WicImageDecoder decoder(bytes, policy);
    decoder.validate_decoded_frame();
    if (policy.checkpoint) { policy.checkpoint(); }
    return decoder.info();
}

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unsupported_video();
}

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    WicImageDecoder decoder(bytes, policy);
    Image image = decoder.decode_rgb();
    if (policy.checkpoint) { policy.checkpoint(); }
    return image;
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unsupported_video();
}

} // namespace ninfer::media::decode
