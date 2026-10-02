namespace {

void validate_snapshot_tokens(std::span<const TokenId> tokens) {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::invalid_argument("context-cache token is outside the target token domain");
        }
    }
}

[[nodiscard]] std::uint8_t snapshot_residency_flags(bool device, bool host) {
    const std::uint8_t flags = static_cast<std::uint8_t>((device ? 1U : 0U) |
                                                         (host ? 2U : 0U));
    if (flags == 0) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache source object has no complete physical replica");
    }
    return flags;
}

void zero_uncommitted_kv_columns(std::span<std::byte> payload, const HostKVPageLayout& layout,
                                std::uint32_t committed_columns) {
    if (committed_columns > layout.geometry.page_tokens ||
        payload.size() != layout.page_stride || layout.planes.size() != layout.geometry.planes.size()) {
        throw std::invalid_argument("context-cache KV page staging geometry is invalid");
    }
    for (std::size_t plane_index = 0; plane_index < layout.planes.size(); ++plane_index) {
        const HostKVPlaneLayout& host = layout.planes[plane_index];
        const KVPlaneGeometry& geometry = layout.geometry.planes[plane_index];
        const std::size_t element_bytes = dtype_size(geometry.dtype);
        const std::size_t token_bytes =
            static_cast<std::size_t>(geometry.leading_extent) * element_bytes;
        for (std::int32_t head = 0; head < geometry.head_extent; ++head) {
            const std::size_t head_offset =
                host.offset + static_cast<std::size_t>(head) * host.head_payload_bytes;
            for (std::uint32_t token = committed_columns; token < layout.geometry.page_tokens;
                 ++token) {
                std::memset(payload.data() + head_offset + static_cast<std::size_t>(token) * token_bytes,
                            0, token_bytes);
            }
        }
    }
}

[[nodiscard]] bool canonical_kv_page_bytes(std::span<const std::byte> payload,
                                           const HostKVPageLayout& layout,
                                           std::uint32_t committed_columns) {
    if (payload.size() != layout.page_stride ||
        committed_columns > layout.geometry.page_tokens ||
        layout.planes.size() != layout.geometry.planes.size()) {
        return false;
    }
    std::size_t prior_end = 0;
    for (std::size_t plane_index = 0; plane_index < layout.planes.size(); ++plane_index) {
        const HostKVPlaneLayout& host = layout.planes[plane_index];
        const KVPlaneGeometry& geometry = layout.geometry.planes[plane_index];
        if (host.offset < prior_end || host.offset > payload.size() ||
            host.page_payload_bytes > payload.size() - host.offset) {
            return false;
        }
        const auto zero_range = [&](std::size_t begin, std::size_t count) {
            return std::all_of(payload.begin() + static_cast<std::ptrdiff_t>(begin),
                               payload.begin() + static_cast<std::ptrdiff_t>(begin + count),
                               [](std::byte value) { return value == std::byte{0}; });
        };
        if (host.offset > prior_end && !zero_range(prior_end, host.offset - prior_end)) {
            return false;
        }
        const std::size_t token_bytes =
            static_cast<std::size_t>(geometry.leading_extent) * dtype_size(geometry.dtype);
        for (std::int32_t head = 0; head < geometry.head_extent; ++head) {
            const std::size_t head_offset =
                host.offset + static_cast<std::size_t>(head) * host.head_payload_bytes;
            const std::size_t tail_begin =
                head_offset + static_cast<std::size_t>(committed_columns) * token_bytes;
            const std::size_t tail_bytes =
                static_cast<std::size_t>(layout.geometry.page_tokens - committed_columns) *
                token_bytes;
            if (!zero_range(tail_begin, tail_bytes)) { return false; }
        }
        prior_end = host.offset + host.page_payload_bytes;
    }
    return prior_end <= payload.size() &&
           std::all_of(payload.begin() + static_cast<std::ptrdiff_t>(prior_end), payload.end(),
                       [](std::byte value) { return value == std::byte{0}; });
}

[[nodiscard]] bool canonical_state_image_bytes(
    std::span<const std::byte> payload, const qwen3_6::StateImageHostLayout& layout) {
    if (payload.size() != layout.image_bytes) { return false; }
    std::vector<std::pair<std::size_t, std::size_t>> regions;
    regions.reserve(5);
    regions.emplace_back(layout.linear_conv.offset, layout.linear_conv.bytes);
    regions.emplace_back(layout.linear_recurrent.offset, layout.linear_recurrent.bytes);
    regions.emplace_back(layout.continuation_hidden.offset, layout.continuation_hidden.bytes);
    if (layout.dflash_local_k) {
        regions.emplace_back(layout.dflash_local_k->offset, layout.dflash_local_k->bytes);
    }
    if (layout.dflash_local_v) {
        regions.emplace_back(layout.dflash_local_v->offset, layout.dflash_local_v->bytes);
    }
    std::sort(regions.begin(), regions.end());
    std::size_t prior_end = 0;
    for (const auto& [offset, bytes] : regions) {
        if (offset < prior_end || offset > payload.size() || bytes > payload.size() - offset) {
            return false;
        }
        if (!std::all_of(payload.begin() + static_cast<std::ptrdiff_t>(prior_end),
                         payload.begin() + static_cast<std::ptrdiff_t>(offset),
                         [](std::byte value) { return value == std::byte{0}; })) {
            return false;
        }
        prior_end = offset + bytes;
    }
    return std::all_of(payload.begin() + static_cast<std::ptrdiff_t>(prior_end), payload.end(),
                       [](std::byte value) { return value == std::byte{0}; });
}

[[nodiscard]] ContextCacheExportState& require_export_session(
    const ProgramImplCore& program, const std::shared_ptr<void>& opaque) {
    if (!opaque) { throw std::invalid_argument("context-cache export session is empty"); }
    auto& session = *static_cast<ContextCacheExportState*>(opaque.get());
    if (session.owner != &program || session.closed) {
        throw std::invalid_argument("context-cache export session is stale or foreign");
    }
    return session;
}

[[nodiscard]] ContextCacheImportState& require_import_session(
    const ProgramImplCore& program, const std::shared_ptr<void>& opaque) {
    if (!opaque) { throw std::invalid_argument("context-cache import session is empty"); }
    auto& session = *static_cast<ContextCacheImportState*>(opaque.get());
    if (session.owner != &program || session.committed || session.rollback_failed) {
        throw std::invalid_argument("context-cache import session is stale or foreign");
    }
    return session;
}

void increment_snapshot_reference(std::uint32_t& value, const char* label) {
    if (value == std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(label);
    }
    ++value;
}

[[nodiscard]] std::uint64_t reserve_export_link(ContextCacheExportState& session,
                                                SnapshotObjectKey key,
                                                bool& is_new) {
    if (const auto found = session.object_ids.find(key); found != session.object_ids.end()) {
        is_new = false;
        return found->second;
    }
    if (session.next_link_id == 0 || session.next_link_id == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("context-cache archive-local object ID is exhausted");
    }
    is_new = true;
    return session.next_link_id;
}

void publish_export_link(ContextCacheExportState& session, SnapshotObjectKey key,
                         std::uint64_t link_id) {
    const auto [iterator, inserted] = session.object_ids.emplace(key, link_id);
    if (!inserted || iterator->second != link_id || session.next_link_id != link_id) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache export object-link topology changed during serialization");
    }
    ++session.next_link_id;
}

void write_state_link(ProgramImplCore& program, ContextCacheExportState& session,
                      StateImageHandle state, ninfer::ContextCacheWriter& writer) {
    if (!program.state_store || !program.state_store->valid(state)) {
        throw ninfer::ContextCacheOwnershipError("context-cache StateImage handle is stale");
    }
    const SnapshotObjectKey key{SnapshotObjectKind::StateImage,
                                program.state_store->archive_key(state)};
    bool is_new = false;
    const std::uint64_t link_id = reserve_export_link(session, key, is_new);
    auto* const counter = dynamic_cast<CountingContextCacheWriter*>(&writer);
    std::optional<PinnedHostBuffer> staging;
    std::uint8_t flags = 0;
    std::size_t payload_bytes = 0;
    if (is_new) {
        const StateReplicaResidency residency = program.state_store->residency(state);
        flags = snapshot_residency_flags(residency == StateReplicaResidency::DeviceOnly ||
                                             residency == StateReplicaResidency::Both,
                                         residency == StateReplicaResidency::HostOnly ||
                                             residency == StateReplicaResidency::Both);
        payload_bytes = program.state_images->host_layout().image_bytes;
        if (counter == nullptr) {
            staging.emplace(payload_bytes);
            auto* data = static_cast<std::byte*>(staging->data());
            try {
                program.state_store->copy_snapshot_payload(
                    state, std::span<std::byte>(data, staging->size()), program.device.stream);
            } catch (const ninfer::ContextCacheOwnershipError&) {
                try {
                    program.device.synchronize();
                } catch (...) {}
                throw;
            } catch (...) {
                try {
                    program.device.synchronize();
                } catch (...) {}
                throw ninfer::ContextCacheOwnershipError(
                    "context-cache StateImage D2H snapshot failed");
            }
        }
    }
    write_u8(writer, static_cast<std::uint8_t>(SnapshotObjectKind::StateImage));
    write_u64(writer, link_id);
    write_bool(writer, is_new);
    if (!is_new) { return; }
    write_u8(writer, flags);
    write_u64(writer, payload_bytes);
    if (counter != nullptr) {
        counter->add(payload_bytes);
    } else {
        write_payload(writer, std::span<const std::byte>(
                                  static_cast<const std::byte*>(staging->data()),
                                  staging->size()));
    }
    publish_export_link(session, key, link_id);
}

void write_kv_link(ProgramImplCore& program, ContextCacheExportState& session,
                   LogicalKVPageStore& pages, LogicalKVPageHandle page,
                   SnapshotObjectKind kind, ninfer::ContextCacheWriter& writer) {
    if ((kind != SnapshotObjectKind::MainKVPage && kind != SnapshotObjectKind::BackendKVPage) ||
        !pages.stable_for_snapshot(page)) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache KV source page is not a stable immutable page");
    }
    const SnapshotObjectKey key{kind, pages.archive_key(page)};
    bool is_new = false;
    const std::uint64_t link_id = reserve_export_link(session, key, is_new);
    auto* const counter = dynamic_cast<CountingContextCacheWriter*>(&writer);
    std::optional<PinnedHostBuffer> staging;
    std::uint8_t flags = 0;
    const std::uint32_t columns = pages.committed_columns(page);
    const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());
    if (is_new) {
        const bool device_replica = pages.device_resident(page);
        const bool host_replica = pages.host_resident(page);
        flags = snapshot_residency_flags(device_replica, host_replica);
        if (counter != nullptr) {
            // The dry-run sizes the owner frame without issuing duplicate device-to-host traffic.
        } else {
            staging.emplace(layout.page_stride);
            auto* data = static_cast<std::byte*>(staging->data());
            std::span<std::byte> payload(data, staging->size());
            if (host_replica) {
            if (program.host_kv_extents == nullptr) {
                throw ninfer::ContextCacheOwnershipError(
                    "context-cache KV Host replica has no owning extent store");
            }
            const HostKVPageReplica& replica = pages.host_replica(page);
            if (!program.host_kv_extents->valid(replica.extent) ||
                replica.content_epoch != pages.content_epoch(page) ||
                replica.committed_columns != columns) {
                throw ninfer::ContextCacheOwnershipError(
                    "context-cache KV Host replica is stale or incomplete");
            }
            const HostKVAllocationConstView page_view =
                program.host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
            if (page_view.layout().geometry != pages.physical_pool().geometry() ||
                page_view.layout().page_stride != layout.page_stride) {
                throw ninfer::ContextCacheOwnershipError(
                    "context-cache KV Host layout does not match its page pool");
            }
                std::fill(payload.begin(), payload.end(), std::byte{0});
                for (const HostKVPlaneLayout& plane : layout.planes) {
                    std::memcpy(payload.data() + plane.offset, page_view.data() + plane.offset,
                                plane.page_payload_bytes);
                }
            } else {
                try {
                    pages.physical_pool().copy_page_to_host(pages.physical(page), payload,
                                                            program.device.stream);
                    program.device.synchronize();
                } catch (...) {
                    try {
                        program.device.synchronize();
                    } catch (...) {}
                    throw ninfer::ContextCacheOwnershipError(
                        "context-cache KV page D2H snapshot failed");
                }
            }
            zero_uncommitted_kv_columns(payload, layout, columns);
        }
    }
    write_u8(writer, static_cast<std::uint8_t>(kind));
    write_u64(writer, link_id);
    write_bool(writer, is_new);
    if (!is_new) { return; }
    write_u8(writer, flags);
    write_u32(writer, columns);
    write_u64(writer, layout.page_stride);
    if (counter != nullptr) {
        counter->add(layout.page_stride);
    } else {
        write_payload(writer, std::span<const std::byte>(
                                  static_cast<const std::byte*>(staging->data()),
                                  staging->size()));
    }
    publish_export_link(session, key, link_id);
}

void write_kv_address(ProgramImplCore& program, ContextCacheExportState& session,
                      KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                      KVAddressSpaceHandle address, SnapshotObjectKind kind,
                      ninfer::ContextCacheWriter& writer) {
    if (!addresses.valid(address) || addresses.active(address) ||
        addresses.bound_row(address) >= 0) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache address space is active or stale");
    }
    const std::uint32_t frontier = addresses.committed_frontier(address);
    const std::uint32_t mapped = addresses.mapped_pages(address);
    if (mapped != kv_pages_for_frontier(frontier)) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache address space has untrimmed or incomplete pages");
    }
    write_u32(writer, frontier);
    write_u32(writer, addresses.checkpoint_frontier(address));
    write_u32(writer, mapped);
    for (std::uint32_t page_index = 0; page_index < mapped; ++page_index) {
        const LogicalKVPageHandle page = addresses.logical_page(address, page_index);
        write_kv_link(program, session, pages, page, kind, writer);
    }
}

[[nodiscard]] std::uint32_t page_count_for_frontier(std::uint32_t frontier) noexcept {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

struct SnapshotRequiredFrontiers {
    std::uint32_t main = 0;
    std::uint32_t backend = 0;
};

void include_required_frontiers(SnapshotRequiredFrontiers& required,
                                std::uint32_t main_frontier,
                                std::uint32_t backend_frontier) noexcept {
    required.main = std::max(required.main, main_frontier);
    required.backend = std::max(required.backend, backend_frontier);
}

[[nodiscard]] SnapshotRequiredFrontiers private_required_frontiers(
    const SequenceState& sequence, bool mtp) noexcept {
    SnapshotRequiredFrontiers required;
    const auto include_checkpoint = [&](std::uint32_t frontier) {
        include_required_frontiers(required, frontier,
                                   mtp && frontier != 0 ? frontier - 1U : 0U);
    };
    if (sequence.endpoint_valid) { include_checkpoint(sequence.execution_frontier); }
    if (sequence.rewrite_checkpoint.valid) {
        include_checkpoint(sequence.rewrite_checkpoint.frontier);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        include_checkpoint(anchor.frontier);
    }
    return required;
}

[[nodiscard]] SnapshotRequiredFrontiers private_required_frontiers(
    const qwen3_6::ContinuationSummary& summary) noexcept {
    SnapshotRequiredFrontiers required;
    const auto include_checkpoint = [&](const qwen3_6::CheckpointSummary& checkpoint) {
        include_required_frontiers(required, checkpoint.required_kv.main_frontier,
                                   checkpoint.required_kv.backend_frontier);
    };
    if (summary.endpoint) { include_checkpoint(*summary.endpoint); }
    if (summary.rewrite) { include_checkpoint(*summary.rewrite); }
    for (const qwen3_6::CheckpointSummary& anchor : summary.long_anchors) {
        include_checkpoint(anchor);
    }
    return required;
}

void write_private_owner_record(ProgramImplCore& program, ContextCacheExportState& session,
                                const SequenceState& sequence,
                                ninfer::ContextCacheWriter& writer) {
    if (sequence.endpoint_valid && sequence.execution_frontier == 0) {
        throw ninfer::ContextCacheOwnershipError(
            "private endpoint has an empty execution frontier");
    }
    if (sequence.endpoint_valid &&
        (!program.state_store->valid(sequence.state.read) ||
         sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
         sequence.state.read_ownership != StateReadOwnership::Primary ||
         program.state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable)) {
        throw ninfer::ContextCacheOwnershipError(
            "private endpoint StateImage is not a stable immutable checkpoint");
    }
    if (sequence.rewrite_checkpoint.valid != sequence.rewrite_state.has_value() ||
        (sequence.rewrite_state &&
         (!program.state_store->valid(*sequence.rewrite_state) ||
          program.state_store->role(*sequence.rewrite_state) !=
              StateImageRole::CheckpointImmutable)) ||
        sequence.reserved_state || sequence.state.fork_pending ||
        !sequence.shared_prefix_references.empty() || !sequence.kv ||
        sequence.ledger_frontier != sequence.ledger.size() ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.ledger_frontier > Variant::maximum_context ||
        sequence.execution_frontier > program.capacity ||
        sequence.execution_frontier > sequence.ledger_frontier ||
        sequence.ledger_frontier - sequence.execution_frontier > 1U ||
        sequence.text_kv_valid != program.text_kv_addresses->committed_frontier(sequence.kv->text) ||
        sequence.text_kv_valid != sequence.execution_frontier ||
        (sequence.endpoint_valid && !sequence.tail_hidden_valid) ||
        sequence.rebuild_tail_begin > sequence.execution_frontier ||
        sequence.rebuild_work.tokens != sequence.execution_frontier ||
        sequence.mtp_draft_count > sequence.mtp_drafts.size() ||
        sequence.state.read_ownership != StateReadOwnership::Primary ||
        sequence.dflash_context_frontier != 0) {
        throw ninfer::ContextCacheOwnershipError(
            "private continuation is not at a complete committed snapshot boundary");
    }
    if (!sequence.endpoint_valid &&
        (program.state_store->valid(sequence.state.read) ||
         program.state_store->valid(sequence.state.write))) {
        throw ninfer::ContextCacheOwnershipError(
            "private continuation has a StateImage without an endpoint");
    }
    if (sequence.rewrite_checkpoint.valid &&
        (sequence.rewrite_checkpoint.frontier == 0 ||
         sequence.rewrite_checkpoint.frontier > sequence.execution_frontier ||
         sequence.rewrite_checkpoint.rebuild_work.tokens !=
             sequence.rewrite_checkpoint.frontier)) {
        throw ninfer::ContextCacheOwnershipError("private rewrite checkpoint is inconsistent");
    }
    try {
        validate_long_anchor_ordinals(sequence.long_anchors,
                                      program.context_cache.max_long_anchors_per_continuation.value_or(0));
    } catch (...) {
        throw ninfer::ContextCacheOwnershipError("private long-anchor set is inconsistent");
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (!program.state_store->valid(anchor.state) ||
            program.state_store->role(anchor.state) != StateImageRole::CheckpointImmutable ||
            anchor.frontier == 0 || anchor.frontier > sequence.execution_frontier ||
            anchor.rebuild_work.tokens != anchor.frontier) {
            throw ninfer::ContextCacheOwnershipError("private long anchor is inconsistent");
        }
    }
    if (program.text_kv_addresses->active(sequence.kv->text) ||
        sequence.text_kv_valid != program.text_kv_addresses->committed_frontier(sequence.kv->text) ||
        program.text_kv_addresses->mapped_pages(sequence.kv->text) !=
            kv_pages_for_frontier(sequence.text_kv_valid)) {
        throw ninfer::ContextCacheOwnershipError("private Main KV address is not stable");
    }
    if (program.speculative_backend == SpeculativeBackend::Mtp) {
        if (!sequence.kv->backend || !program.backend_kv_addresses || !program.backend_kv_pages ||
            program.backend_kv_addresses->active(*sequence.kv->backend) ||
            sequence.mtp_kv_valid !=
                program.backend_kv_addresses->committed_frontier(*sequence.kv->backend) ||
            program.backend_kv_addresses->mapped_pages(*sequence.kv->backend) !=
                kv_pages_for_frontier(sequence.mtp_kv_valid)) {
            throw ninfer::ContextCacheOwnershipError("private MTP KV address is not stable");
        }
    } else if (sequence.kv->backend || sequence.mtp_kv_valid != 0 ||
               sequence.mtp_draft_count != 0) {
        throw ninfer::ContextCacheOwnershipError("ordinary continuation retains MTP state");
    }
    const SnapshotRequiredFrontiers required = private_required_frontiers(
        sequence, program.speculative_backend == SpeculativeBackend::Mtp);
    if (program.text_kv_addresses->checkpoint_frontier(sequence.kv->text) < required.main ||
        (sequence.kv->backend &&
         program.backend_kv_addresses->checkpoint_frontier(*sequence.kv->backend) <
             required.backend)) {
        throw ninfer::ContextCacheOwnershipError(
            "private KV protection is below its checkpoint requirement");
    }

    write_target_header(writer, program, ninfer::ContextCacheOwnerKind::PrivateContinuation);
    write_u32(writer, sequence.execution_frontier);
    write_u32(writer, sequence.ledger_frontier);
    write_u32(writer, sequence.text_kv_valid);
    write_u32(writer, sequence.mtp_kv_valid);
    write_u32(writer, sequence.dflash_context_frontier);
    write_i32(writer, sequence.rope_delta);
    write_bool(writer, sequence.endpoint_valid);
    write_u8(writer, static_cast<std::uint8_t>(sequence.state.read_ownership));
    write_bool(writer, sequence.tail_hidden_valid);
    write_prefill_work(writer, sequence.rebuild_work);
    write_u32(writer, sequence.rebuild_tail_begin);
    write_u32(writer, sequence.mtp_draft_count);
    write_little_endian_span(writer, std::span<const TokenId>(sequence.mtp_drafts.data(),
                                                             sequence.mtp_draft_count));
    write_u32(writer, sequence.ledger_frontier);
    write_token_ledger(writer, sequence.ledger);
    write_prefix_identity(writer, sequence.prefix_identity);
    write_bool(writer, sequence.rewrite_checkpoint.valid);
    if (sequence.rewrite_checkpoint.valid) {
        write_u8(writer, static_cast<std::uint8_t>(sequence.rewrite_checkpoint.kind));
        write_u32(writer, sequence.rewrite_checkpoint.frontier);
        write_prefill_work(writer, sequence.rewrite_checkpoint.rebuild_work);
    }
    write_u32(writer, static_cast<std::uint32_t>(sequence.long_anchors.size()));
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        write_u32(writer, anchor.ordinal);
        write_u32(writer, anchor.frontier);
        write_prefill_work(writer, anchor.rebuild_work);
    }
    if (sequence.endpoint_valid) { write_state_link(program, session, sequence.state.read, writer); }
    if (sequence.rewrite_state) { write_state_link(program, session, *sequence.rewrite_state, writer); }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        write_state_link(program, session, anchor.state, writer);
    }
    write_kv_address(program, session, *program.text_kv_addresses, *program.text_kv_pages,
                     sequence.kv->text, SnapshotObjectKind::MainKVPage, writer);
    write_bool(writer, sequence.kv->backend.has_value());
    if (sequence.kv->backend) {
        write_kv_address(program, session, *program.backend_kv_addresses,
                         *program.backend_kv_pages, *sequence.kv->backend,
                         SnapshotObjectKind::BackendKVPage, writer);
    }
}

void write_shared_owner_record(ProgramImplCore& program, ContextCacheExportState& session,
                               const SharedPrefixState& shared,
                               ninfer::ContextCacheWriter& writer) {
    if (!shared.kv || !shared.identity || shared.frontier == 0 ||
        shared.frontier > program.capacity || shared.active_references != 0 ||
        !program.state_store->valid(shared.state) ||
        program.state_store->role(shared.state) != StateImageRole::CheckpointImmutable ||
        !shared.identity->backing || shared.identity->shortlist_key.frontier != shared.frontier ||
        shared.rebuild_work.tokens != shared.frontier ||
        !program.text_kv_addresses->valid(shared.kv->text) ||
        program.text_kv_addresses->active(shared.kv->text) ||
        program.text_kv_addresses->committed_frontier(shared.kv->text) != shared.frontier ||
        program.text_kv_addresses->mapped_pages(shared.kv->text) !=
            kv_pages_for_frontier(shared.frontier) ||
        program.text_kv_addresses->checkpoint_frontier(shared.kv->text) < shared.frontier) {
        throw ninfer::ContextCacheOwnershipError("shared prefix is not a stable checkpoint");
    }
    if (program.speculative_backend == SpeculativeBackend::Mtp) {
        if (!shared.kv->backend || !program.backend_kv_addresses || !program.backend_kv_pages ||
            program.backend_kv_addresses->active(*shared.kv->backend) ||
            program.backend_kv_addresses->committed_frontier(*shared.kv->backend) <
                shared.backend_frontier ||
            program.backend_kv_addresses->committed_frontier(*shared.kv->backend) >
                shared.frontier ||
            program.backend_kv_addresses->mapped_pages(*shared.kv->backend) !=
                kv_pages_for_frontier(program.backend_kv_addresses->committed_frontier(*shared.kv->backend)) ||
            program.backend_kv_addresses->checkpoint_frontier(*shared.kv->backend) <
                shared.backend_frontier ||
            shared.backend_frontier != shared.frontier - 1U) {
            throw ninfer::ContextCacheOwnershipError("shared-prefix MTP KV is incomplete");
        }
    } else if (shared.kv->backend || shared.backend_frontier != 0) {
        throw ninfer::ContextCacheOwnershipError("ordinary shared prefix has backend KV state");
    }
    const std::span<const TokenId> ledger = shared.identity->ledger();
    const auto* source_identity = shared.identity->prefix_identity();
    if (source_identity == nullptr || ledger.size() != shared.frontier ||
        source_identity->size() < shared.frontier) {
        throw ninfer::ContextCacheOwnershipError("shared-prefix exact identity is incomplete");
    }
    qwen3_6::detail::ResidentPrefixIdentity prefix = *source_identity;
    try {
        prefix.truncate(shared.frontier);
    } catch (...) {
        throw ninfer::ContextCacheOwnershipError("shared-prefix exact identity cannot be truncated");
    }

    write_target_header(writer, program, ninfer::ContextCacheOwnerKind::SharedPrefix);
    write_u32(writer, shared.frontier);
    write_u32(writer, shared.backend_frontier);
    write_i32(writer, shared.rope_delta);
    write_bool(writer, shared.tail_hidden_valid);
    write_prefill_work(writer, shared.rebuild_work);
    write_u64(writer, shared.identity->shortlist_key.digests[0]);
    write_u64(writer, shared.identity->shortlist_key.digests[1]);
    write_u32(writer, shared.identity->shortlist_key.identity_tag);
    write_u32(writer, static_cast<std::uint32_t>(ledger.size()));
    write_token_ledger(writer, ledger);
    write_prefix_identity(writer, prefix);
    write_state_link(program, session, shared.state, writer);
    write_kv_address(program, session, *program.text_kv_addresses, *program.text_kv_pages,
                     shared.kv->text, SnapshotObjectKind::MainKVPage, writer);
    write_bool(writer, shared.kv->backend.has_value());
    if (shared.kv->backend) {
        write_kv_address(program, session, *program.backend_kv_addresses,
                         *program.backend_kv_pages, *shared.kv->backend,
                         SnapshotObjectKind::BackendKVPage, writer);
    }
}

[[nodiscard]] std::uint32_t snapshot_identity_tag(const ProgramImplCore& program) noexcept {
    return static_cast<std::uint32_t>(program.speculative_backend) |
           (static_cast<std::uint32_t>(program.proposal_head) << 8U) |
           (static_cast<std::uint32_t>(program.kv_storage) << 16U);
}

[[nodiscard]] runtime::ReplicaResidency snapshot_state_residency(
    StateImageStore& store, StateImageHandle handle) {
    switch (store.residency(handle)) {
    case StateReplicaResidency::DeviceOnly: return runtime::ReplicaResidency::DeviceOnly;
    case StateReplicaResidency::HostOnly: return runtime::ReplicaResidency::HostOnly;
    case StateReplicaResidency::Both: return runtime::ReplicaResidency::Both;
    case StateReplicaResidency::None:
    default: throw std::invalid_argument("context-cache StateImage has no physical replica");
    }
}

[[nodiscard]] qwen3_6::CheckpointSummary make_snapshot_checkpoint_summary(
    const ProgramImplCore& program, runtime::CheckpointRef checkpoint,
    runtime::CheckpointScope scope, StateImageHandle state,
    std::array<std::uint64_t, 2> digests, std::uint32_t identity_tag,
    runtime::PrefillWork rebuild_work, std::uint32_t main_frontier,
    std::uint32_t backend_frontier) {
    if (!program.state_store->valid(state) || main_frontier > program.capacity ||
        backend_frontier > program.capacity || backend_frontier > main_frontier ||
        checkpoint.frontier != main_frontier) {
        throw std::invalid_argument("context-cache checkpoint owner references are inconsistent");
    }
    return qwen3_6::CheckpointSummary{
        .ref = checkpoint,
        .scope = scope,
        .shortlist_key = qwen3_6::PrefixShortlistKey{
            .digests = digests,
            .frontier = checkpoint.frontier,
            .identity_tag = identity_tag,
        },
        .state_residency = snapshot_state_residency(*program.state_store, state),
        .required_kv = qwen3_6::TargetKVRequirement{
            .main_frontier = main_frontier,
            .backend_frontier = backend_frontier,
            .main_pages = page_count_for_frontier(main_frontier),
            .backend_pages = page_count_for_frontier(backend_frontier),
        },
        .rebuild_work = validated_rebuild_work(rebuild_work, checkpoint.frontier),
    };
}

[[nodiscard]] std::size_t lookup_import_link(const ContextCacheImportState& session,
                                             std::uint64_t link_id,
                                             SnapshotObjectKind expected_kind) {
    const auto found = session.object_indices.find(link_id);
    if (link_id == 0 || found == session.object_indices.end() ||
        found->second >= session.objects.size() ||
        session.objects[found->second].link_id != link_id ||
        session.objects[found->second].kind != expected_kind) {
        throw std::invalid_argument("context-cache object link is unresolved or has the wrong type");
    }
    return found->second;
}

enum class SnapshotStateReferenceRole : std::uint8_t {
    Primary,
    Checkpoint,
    External,
};

[[nodiscard]] StateImageHandle read_state_link(ProgramImplCore& program,
                                               ContextCacheImportState& session,
                                               ContextCacheReader& reader,
                                               SnapshotStateReferenceRole role) {
    const std::uint8_t encoded_kind = read_u8(reader);
    const std::uint64_t link_id = read_u64(reader);
    const bool is_new = read_bool(reader);
    if (encoded_kind != static_cast<std::uint8_t>(SnapshotObjectKind::StateImage) || link_id == 0) {
        throw std::invalid_argument("context-cache StateImage object link is invalid");
    }
    std::size_t object_index = 0;
    if (!is_new) {
        object_index = lookup_import_link(session, link_id, SnapshotObjectKind::StateImage);
    } else {
        if (link_id != session.next_link_id ||
            link_id == std::numeric_limits<std::uint64_t>::max() ||
            session.object_indices.contains(link_id)) {
            throw std::invalid_argument("context-cache StateImage archive-local ID is invalid");
        }
        const std::uint8_t flags = read_u8(reader);
        if (flags == 0 || (flags & ~std::uint8_t{3}) != 0) {
            throw std::invalid_argument("context-cache StateImage residency flags are invalid");
        }
        const bool device_replica = (flags & 1U) != 0;
        const bool host_replica = (flags & 2U) != 0;
        const std::uint64_t payload_bytes = read_u64(reader);
        const std::size_t expected_bytes = program.state_images->host_layout().image_bytes;
        if (payload_bytes != expected_bytes || payload_bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("context-cache StateImage payload size does not match");
        }
        PinnedHostBuffer staging(static_cast<std::size_t>(payload_bytes));
        std::span<std::byte> payload(static_cast<std::byte*>(staging.data()), staging.size());
        read_raw(reader, payload);
        if (!canonical_state_image_bytes(payload, program.state_images->host_layout())) {
            throw std::invalid_argument("context-cache StateImage padding is not canonical");
        }

        session.objects.emplace_back();
        object_index = session.objects.size() - 1U;
        ContextCacheImportObject& object = session.objects.back();
        object.link_id = link_id;
        object.kind = SnapshotObjectKind::StateImage;
        object.device_replica = device_replica;
        object.host_replica = host_replica;
        const std::optional<StateImageHandle> destination =
            program.state_store->reserve_import_destination(device_replica);
        if (!destination) { throw std::bad_alloc(); }
        object.state = *destination;
        try {
            program.state_store->restore_import_payload(object.state, payload, device_replica,
                                                        host_replica, program.device.stream);
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            try {
                program.device.synchronize();
            } catch (...) {
                throw ninfer::ContextCacheOwnershipError(
                    "context-cache StateImage H2D staging could not be synchronized");
            }
            std::rethrow_exception(failure);
        }
        session.object_indices.emplace(link_id, object_index);
        ++session.next_link_id;
    }
    ContextCacheImportObject& object = session.objects[object_index];
    if (role == SnapshotStateReferenceRole::Primary) {
        increment_snapshot_reference(object.primary_owners,
                                     "context-cache StateImage primary-owner count overflows");
        if (object.primary_owners != 1) {
            throw std::invalid_argument("context-cache StateImage has multiple primary owners");
        }
    } else if (role == SnapshotStateReferenceRole::Checkpoint) {
        increment_snapshot_reference(object.state_references,
                                     "context-cache StateImage reference count overflows");
    }
    return object.state;
}

void ensure_import_page_reservation(ProgramImplCore& program, ContextCacheImportState& session,
                                    bool backend) {
    LogicalKVPageStore* pages = backend ? program.backend_kv_pages.get()
                                        : program.text_kv_pages.get();
    if (pages == nullptr) { throw std::invalid_argument("context-cache page pool is unavailable"); }
    DeviceKVPageReservation& reservation =
        backend ? session.backend_reservation : session.main_reservation;
    if (!reservation.valid()) { reservation = pages->physical_pool().make_empty_reservation(); }
    if (reservation.pages() == std::numeric_limits<std::uint32_t>::max()) {
        throw std::bad_alloc();
    }
    const std::uint32_t required = reservation.pages() + 1U;
    if (!pages->physical_pool().can_resize_reservation(reservation, required)) {
        throw std::bad_alloc();
    }
    pages->physical_pool().resize_reservation(reservation, required);
}

[[nodiscard]] std::uint64_t read_kv_link(ProgramImplCore& program,
                                         ContextCacheImportState& session,
                                         LogicalKVPageStore& pages, bool backend,
                                         ContextCacheReader& reader) {
    const SnapshotObjectKind expected_kind =
        backend ? SnapshotObjectKind::BackendKVPage : SnapshotObjectKind::MainKVPage;
    const std::uint8_t encoded_kind = read_u8(reader);
    const std::uint64_t link_id = read_u64(reader);
    const bool is_new = read_bool(reader);
    if (encoded_kind != static_cast<std::uint8_t>(expected_kind) || link_id == 0) {
        throw std::invalid_argument("context-cache KV page object link is invalid");
    }
    std::size_t object_index = 0;
    if (!is_new) {
        object_index = lookup_import_link(session, link_id, expected_kind);
    } else {
        if (link_id != session.next_link_id ||
            link_id == std::numeric_limits<std::uint64_t>::max() ||
            session.object_indices.contains(link_id)) {
            throw std::invalid_argument("context-cache KV archive-local ID is invalid");
        }
        const std::uint8_t flags = read_u8(reader);
        const std::uint32_t columns = read_u32(reader);
        const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());
        const std::uint64_t payload_bytes = read_u64(reader);
        if (flags == 0 || (flags & ~std::uint8_t{3}) != 0 ||
            columns == 0 || columns > layout.geometry.page_tokens ||
            payload_bytes != layout.page_stride || payload_bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("context-cache KV page shape or residency is invalid");
        }
        const bool device_replica = (flags & 1U) != 0;
        const bool host_replica = (flags & 2U) != 0;
        PinnedHostBuffer staging(layout.page_stride);
        std::span<std::byte> payload(static_cast<std::byte*>(staging.data()), staging.size());
        read_raw(reader, payload);
        if (!canonical_kv_page_bytes(payload, layout, columns)) {
            throw std::invalid_argument("context-cache KV page padding or uncommitted tail is not canonical");
        }

        session.objects.emplace_back();
        object_index = session.objects.size() - 1U;
        ContextCacheImportObject& object = session.objects.back();
        object.link_id = link_id;
        object.kind = expected_kind;
        object.committed_columns = columns;
        object.device_replica = device_replica;
        object.host_replica = host_replica;
        DeviceKVPageReservation& reservation =
            backend ? session.backend_reservation : session.main_reservation;
        if (device_replica) {
            ensure_import_page_reservation(program, session, backend);
            object.page = pages.materialize_transfer_destination(reservation, columns);
        } else {
            object.page = pages.materialize_host_import_destination(columns);
        }
        if (host_replica) {
            if (program.host_kv_extents == nullptr || program.host_kv_arena == nullptr) {
                throw ninfer::ContextCacheUnsupported(
                    "context-cache archive requires Host KV replica support");
            }
            std::optional<HostKVAllocation> allocation =
                program.host_kv_extents->allocate_imported_page(pages);
            if (!allocation) { throw std::bad_alloc(); }
            HostKVAllocationView destination = program.host_kv_arena->writable_view(*allocation);
            if (destination.layout().geometry != layout.geometry ||
                destination.layout().page_stride != layout.page_stride ||
                destination.page_count() != 1) {
                throw std::invalid_argument("context-cache Host KV import layout does not match");
            }
            std::memcpy(destination.data(), payload.data(), payload.size());
            object.host_allocation.emplace(std::move(*allocation));
        }
        if (device_replica) {
            try {
                pages.physical_pool().copy_page_from_host(
                    std::span<const std::byte>(payload.data(), payload.size()),
                    pages.physical(object.page), program.device.stream);
                program.device.synchronize();
            } catch (...) {
                try {
                    program.device.synchronize();
                } catch (...) {}
                throw ninfer::ContextCacheOwnershipError("context-cache KV page H2D import failed");
            }
        }
        session.object_indices.emplace(link_id, object_index);
        ++session.next_link_id;
    }
    ContextCacheImportObject& object = session.objects[object_index];
    increment_snapshot_reference(object.address_references,
                                 "context-cache KV page reference count overflows");
    return link_id;
}

[[nodiscard]] ContextCacheImportAddress read_kv_address(
    ProgramImplCore& program, ContextCacheImportState& session,
    ContextCacheReader& reader, bool backend) {
    ContextCacheImportAddress out;
    out.backend = backend;
    out.frontier = read_u32(reader);
    out.checkpoint_frontier = read_u32(reader);
    const std::uint32_t page_count = read_u32(reader);
    LogicalKVPageStore* pages = backend ? program.backend_kv_pages.get()
                                        : program.text_kv_pages.get();
    KVAddressSpaceStore* addresses = backend ? program.backend_kv_addresses.get()
                                              : program.text_kv_addresses.get();
    if (pages == nullptr || addresses == nullptr || out.frontier > Variant::maximum_context ||
        out.checkpoint_frontier > out.frontier ||
        page_count != page_count_for_frontier(out.frontier) || page_count > pages->capacity()) {
        throw std::invalid_argument("context-cache address-space frontier is invalid");
    }
    out.page_links.reserve(page_count);
    for (std::uint32_t index = 0; index < page_count; ++index) {
        out.page_links.push_back(read_kv_link(program, session, *pages, backend, reader));
    }
    const std::optional<KVAddressSpaceHandle> address = addresses->create_inactive();
    if (!address) { throw std::bad_alloc(); }
    out.address = *address;
    try {
        session.allocated_addresses.push_back(
            ContextCacheStagedAddress{.address = *address, .backend = backend});
    } catch (...) {
        if (!addresses->release(*address)) { throw ninfer::ContextCacheOwnershipError(
            "context-cache rollback could not release a new address space"); }
        throw;
    }
    return out;
}

[[nodiscard]] std::uint32_t reserve_private_owner_slot(ProgramImplCore& program,
                                                       ContextCacheImportState& session) {
    for (std::uint32_t index = 0; index < program.continuation_capacity; ++index) {
        if (program.continuation_slots[index].role != ContinuationSlotRole::Free) { continue; }
        program.continuation_slots[index].role = ContinuationSlotRole::ReservedMaterialization;
        session.private_slot_indices.push_back(index);
        return index;
    }
    throw std::bad_alloc();
}

[[nodiscard]] std::uint32_t reserve_shared_owner_slot(ProgramImplCore& program,
                                                      ContextCacheImportState& session) {
    for (std::uint32_t index = 0; index < program.shared_prefix_capacity; ++index) {
        if (program.shared_prefix_slots[index].role != SharedPrefixSlotRole::Free) { continue; }
        program.shared_prefix_slots[index].role = SharedPrefixSlotRole::ReservedCapture;
        session.shared_slot_indices.push_back(index);
        return index;
    }
    throw std::bad_alloc();
}

[[nodiscard]] qwen3_6::ContinuationSummary make_imported_continuation_summary(
    const ProgramImplCore& program, const SequenceState& sequence) {
    qwen3_6::ContinuationSummary out;
    if (sequence.endpoint_valid) {
        out.endpoint = make_snapshot_checkpoint_summary(
            program,
            runtime::CheckpointRef{.kind = runtime::CheckpointKind::SessionEndpoint,
                                   .frontier = sequence.execution_frontier},
            runtime::CheckpointScope::Private, sequence.state.read,
            sequence.prefix_digests.at(sequence.execution_frontier), snapshot_identity_tag(program),
            sequence.rebuild_work, sequence.execution_frontier,
            program.speculative_backend == SpeculativeBackend::Mtp
                ? sequence.execution_frontier - 1U
                : 0U);
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::invalid_argument("context-cache rewrite checkpoint has no StateImage");
        }
        out.rewrite = make_snapshot_checkpoint_summary(
            program,
            runtime::CheckpointRef{.kind = checkpoint_kind(sequence.rewrite_checkpoint.kind),
                                   .frontier = sequence.rewrite_checkpoint.frontier},
            runtime::CheckpointScope::Private, *sequence.rewrite_state,
            sequence.prefix_digests.at(sequence.rewrite_checkpoint.frontier),
            snapshot_identity_tag(program), sequence.rewrite_checkpoint.rebuild_work,
            sequence.rewrite_checkpoint.frontier,
            program.speculative_backend == SpeculativeBackend::Mtp
                ? sequence.rewrite_checkpoint.frontier - 1U
                : 0U);
    }
    out.long_anchors.reserve(sequence.long_anchors.size());
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        out.long_anchors.push_back(make_snapshot_checkpoint_summary(
            program,
            runtime::CheckpointRef{.kind = runtime::CheckpointKind::LongAnchor,
                                   .frontier = anchor.frontier,
                                   .ordinal = anchor.ordinal},
            runtime::CheckpointScope::Private, anchor.state,
            sequence.prefix_digests.at(anchor.frontier), snapshot_identity_tag(program),
            anchor.rebuild_work, anchor.frontier,
            program.speculative_backend == SpeculativeBackend::Mtp
                ? anchor.frontier - 1U
                : 0U));
    }
    out.active_references = 0;
    if (!out.endpoint && !out.rewrite && out.long_anchors.empty()) {
        throw std::invalid_argument("context-cache private owner has no checkpoint");
    }
    return out;
}

[[nodiscard]] qwen3_6::SharedPrefixSummary make_imported_shared_summary(
    const ProgramImplCore& program, const SharedPrefixState& shared) {
    if (!shared.identity) { throw std::invalid_argument("context-cache shared identity is absent"); }
    return qwen3_6::SharedPrefixSummary{
        .checkpoint = make_snapshot_checkpoint_summary(
            program,
            runtime::CheckpointRef{.kind = runtime::CheckpointKind::SharedStablePrefix,
                                   .frontier = shared.frontier},
            runtime::CheckpointScope::Shared, shared.state, shared.identity->shortlist_key.digests,
            shared.identity->shortlist_key.identity_tag, shared.rebuild_work, shared.frontier,
            shared.backend_frontier),
        .active_references = 0,
    };
}

[[nodiscard]] ContextCacheImportedPrivateOwner read_private_owner_body(
    ProgramImplCore& program, ContextCacheImportState& session,
    ContextCacheReader& reader) {
    ContextCacheImportedPrivateOwner imported;
    SequenceState& sequence = imported.sequence;
    sequence.execution_frontier = read_u32(reader);
    sequence.ledger_frontier = read_u32(reader);
    sequence.text_kv_valid = read_u32(reader);
    sequence.mtp_kv_valid = read_u32(reader);
    sequence.dflash_context_frontier = read_u32(reader);
    sequence.rope_delta = read_i32(reader);
    sequence.endpoint_valid = read_bool(reader);
    const std::uint8_t ownership = read_u8(reader);
    if (ownership > static_cast<std::uint8_t>(StateReadOwnership::ExternalOwner)) {
        throw std::invalid_argument("context-cache StateImage ownership tag is invalid");
    }
    sequence.state.read_ownership = static_cast<StateReadOwnership>(ownership);
    sequence.tail_hidden_valid = read_bool(reader);
    sequence.rebuild_work = read_prefill_work(reader);
    sequence.rebuild_tail_begin = read_u32(reader);
    sequence.mtp_draft_count = read_u32(reader);
    if (sequence.mtp_draft_count > sequence.mtp_drafts.size() ||
        sequence.mtp_draft_count > program.draft_window ||
        (program.speculative_backend != SpeculativeBackend::Mtp &&
         sequence.mtp_draft_count != 0)) {
        throw std::invalid_argument("context-cache MTP proposal checkpoint is invalid");
    }
    read_little_endian_span(
        reader, std::span<TokenId>(sequence.mtp_drafts.data(), sequence.mtp_draft_count));
    validate_snapshot_tokens(
        std::span<const TokenId>(sequence.mtp_drafts.data(), sequence.mtp_draft_count));

    const std::uint32_t ledger_count = read_u32(reader);
    if (ledger_count != sequence.ledger_frontier ||
        sequence.ledger_frontier > Variant::maximum_context ||
        sequence.execution_frontier > program.capacity ||
        sequence.execution_frontier > sequence.ledger_frontier ||
        sequence.ledger_frontier - sequence.execution_frontier > 1U ||
        sequence.text_kv_valid != sequence.execution_frontier ||
        sequence.dflash_context_frontier != 0 ||
        sequence.rebuild_tail_begin > sequence.execution_frontier ||
        sequence.rebuild_work.tokens != sequence.execution_frontier ||
        (sequence.endpoint_valid && !sequence.tail_hidden_valid) ||
        (sequence.endpoint_valid &&
         sequence.state.read_ownership != StateReadOwnership::Primary) ||
        (!sequence.endpoint_valid &&
         sequence.state.read_ownership != StateReadOwnership::Primary)) {
        throw std::invalid_argument("context-cache private continuation frontier is invalid");
    }
    sequence.ledger = read_token_ledger(reader, ledger_count, Variant::maximum_context);
    validate_snapshot_tokens(sequence.ledger);
    sequence.prefix_identity = read_prefix_identity(reader, ledger_count);
    sequence.prefix_digests.assign_snapshot(sequence.ledger, sequence.prefix_identity);

    sequence.rewrite_checkpoint.valid = read_bool(reader);
    if (sequence.rewrite_checkpoint.valid) {
        const std::uint8_t kind = read_u8(reader);
        if (kind > static_cast<std::uint8_t>(RewriteCheckpointKind::ResponseReplay)) {
            throw std::invalid_argument("context-cache rewrite checkpoint kind is invalid");
        }
        sequence.rewrite_checkpoint.kind = static_cast<RewriteCheckpointKind>(kind);
        sequence.rewrite_checkpoint.frontier = read_u32(reader);
        sequence.rewrite_checkpoint.rebuild_work = read_prefill_work(reader);
        if (sequence.rewrite_checkpoint.frontier == 0 ||
            sequence.rewrite_checkpoint.frontier > sequence.execution_frontier ||
            sequence.rewrite_checkpoint.rebuild_work.tokens !=
                sequence.rewrite_checkpoint.frontier) {
            throw std::invalid_argument("context-cache rewrite checkpoint frontier is invalid");
        }
    }
    const std::uint32_t anchor_count = read_count(
        reader, program.context_cache.max_long_anchors_per_continuation.value_or(0),
        "context-cache long-anchor count exceeds configured capacity");
    sequence.long_anchors.reserve(anchor_count);
    for (std::uint32_t index = 0; index < anchor_count; ++index) {
        LongAnchorCheckpoint anchor;
        anchor.ordinal = read_u32(reader);
        anchor.frontier = read_u32(reader);
        anchor.rebuild_work = read_prefill_work(reader);
        if (anchor.frontier == 0 || anchor.frontier > sequence.execution_frontier ||
            anchor.rebuild_work.tokens != anchor.frontier) {
            throw std::invalid_argument("context-cache long-anchor frontier is invalid");
        }
        sequence.long_anchors.push_back(std::move(anchor));
    }
    try {
        validate_long_anchor_ordinals(
            sequence.long_anchors,
            program.context_cache.max_long_anchors_per_continuation.value_or(0));
    } catch (...) {
        throw std::invalid_argument("context-cache long-anchor ordinals are invalid");
    }
    for (const std::uint32_t rewrite_frontier :
         sequence.prefix_identity.rewrite_execution_frontiers()) {
        if (rewrite_frontier > sequence.execution_frontier) {
            throw std::invalid_argument("context-cache rewrite identity exceeds execution frontier");
        }
    }

    if (sequence.endpoint_valid) {
        sequence.state.read = read_state_link(program, session, reader,
                                              SnapshotStateReferenceRole::Primary);
        sequence.state.write = sequence.state.read;
    }
    if (sequence.rewrite_checkpoint.valid) {
        sequence.rewrite_state = read_state_link(
            program, session, reader, SnapshotStateReferenceRole::Checkpoint);
    }
    for (LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        anchor.state = read_state_link(program, session, reader,
                                       SnapshotStateReferenceRole::Checkpoint);
    }
    if (sequence.rewrite_checkpoint.valid != sequence.rewrite_state.has_value()) {
        throw std::invalid_argument("context-cache rewrite state binding is incomplete");
    }

    imported.main_address = read_kv_address(program, session, reader, false);
    const bool has_backend = read_bool(reader);
    const bool expect_backend = program.speculative_backend == SpeculativeBackend::Mtp;
    if (has_backend != expect_backend) {
        throw std::invalid_argument("context-cache private backend address presence does not match");
    }
    if (has_backend) { imported.backend_address = read_kv_address(program, session, reader, true); }
    if (imported.main_address.frontier != sequence.text_kv_valid ||
        imported.main_address.frontier != sequence.execution_frontier ||
        (imported.backend_address.has_value() != expect_backend) ||
        (imported.backend_address &&
         imported.backend_address->frontier != sequence.mtp_kv_valid) ||
        (expect_backend && sequence.mtp_kv_valid > sequence.execution_frontier) ||
        (!expect_backend && sequence.mtp_kv_valid != 0)) {
        throw std::invalid_argument("context-cache private KV frontiers do not match");
    }
    sequence.kv = SequenceKVBundle{.text = imported.main_address.address};
    if (imported.backend_address) { sequence.kv->backend = imported.backend_address->address; }
    imported.summary = make_imported_continuation_summary(program, sequence);
    const SnapshotRequiredFrontiers required = private_required_frontiers(imported.summary);
    if (imported.main_address.checkpoint_frontier < required.main ||
        (imported.backend_address &&
         imported.backend_address->checkpoint_frontier < required.backend)) {
        throw std::invalid_argument(
            "context-cache private protected KV frontier is below its checkpoint requirement");
    }
    return imported;
}

[[nodiscard]] ContextCacheImportedSharedOwner read_shared_owner_body(
    ProgramImplCore& program, ContextCacheImportState& session,
    ContextCacheReader& reader) {
    ContextCacheImportedSharedOwner imported;
    SharedPrefixState& shared = imported.shared;
    shared.frontier = read_u32(reader);
    shared.backend_frontier = read_u32(reader);
    shared.rope_delta = read_i32(reader);
    shared.tail_hidden_valid = read_bool(reader);
    shared.rebuild_work = read_prefill_work(reader);
    const std::array<std::uint64_t, 2> archived_digests{read_u64(reader), read_u64(reader)};
    const std::uint32_t archived_identity_tag = read_u32(reader);
    const std::uint32_t ledger_count = read_u32(reader);
    if (shared.frontier == 0 || shared.frontier > Variant::maximum_context ||
        shared.frontier > program.capacity ||
        ledger_count != shared.frontier ||
        shared.rebuild_work.tokens != shared.frontier) {
        throw std::invalid_argument("context-cache shared-prefix frontier is invalid");
    }
    std::vector<TokenId> ledger =
        read_token_ledger(reader, ledger_count, Variant::maximum_context);
    validate_snapshot_tokens(ledger);
    qwen3_6::detail::ResidentPrefixIdentity prefix = read_prefix_identity(reader, ledger_count);
    qwen3_6::detail::PrefixShortlistDigests digests;
    digests.assign_snapshot(ledger, prefix);
    const std::array<std::uint64_t, 2> computed_digests = digests.at(shared.frontier);
    if (computed_digests != archived_digests ||
        archived_identity_tag != snapshot_identity_tag(program)) {
        throw std::invalid_argument("context-cache shared-prefix shortlist identity is invalid");
    }
    auto restored_identity = std::make_shared<PreparedCaptureIdentity>();
    auto backing = std::make_shared<PreparedCaptureBacking>();
    backing->ledger = std::move(ledger);
    backing->prefix_identity = std::move(prefix);
    restored_identity->backing = std::move(backing);
    restored_identity->shortlist_key = qwen3_6::PrefixShortlistKey{
        .digests = computed_digests,
        .frontier = shared.frontier,
        .identity_tag = archived_identity_tag,
    };
    restored_identity->rebuild_work = shared.rebuild_work;
    shared.identity = std::move(restored_identity);

    shared.state = read_state_link(program, session, reader,
                                   SnapshotStateReferenceRole::Checkpoint);
    imported.main_address = read_kv_address(program, session, reader, false);
    const bool has_backend = read_bool(reader);
    const bool expect_backend = program.speculative_backend == SpeculativeBackend::Mtp;
    if (has_backend != expect_backend) {
        throw std::invalid_argument("context-cache shared backend address presence does not match");
    }
    if (has_backend) { imported.backend_address = read_kv_address(program, session, reader, true); }
    if (imported.main_address.frontier != shared.frontier ||
        imported.main_address.checkpoint_frontier < shared.frontier ||
        (imported.backend_address.has_value() != expect_backend) ||
        (imported.backend_address &&
         (imported.backend_address->frontier < shared.backend_frontier ||
          imported.backend_address->frontier > shared.frontier ||
          imported.backend_address->checkpoint_frontier < shared.backend_frontier)) ||
        (expect_backend && shared.backend_frontier != shared.frontier - 1U) ||
        (!expect_backend && shared.backend_frontier != 0)) {
        throw std::invalid_argument("context-cache shared KV frontiers do not match");
    }
    shared.kv = SequenceKVBundle{.text = imported.main_address.address};
    if (imported.backend_address) { shared.kv->backend = imported.backend_address->address; }
    imported.summary = make_imported_shared_summary(program, shared);
    return imported;
}

void register_import_owner(ProgramImplCore& program, ContextCacheImportState& session,
                           ContextCacheImportedOwner&& owner) {
    if (session.owners.size() >= session.expected_owner_count) {
        throw std::invalid_argument("context-cache archive has too many owners");
    }
    if (std::holds_alternative<ContextCacheImportedPrivateOwner>(owner)) {
        auto& private_owner = std::get<ContextCacheImportedPrivateOwner>(owner);
        private_owner.slot_index = reserve_private_owner_slot(program, session);
        const std::uint32_t slot_index = private_owner.slot_index;
        try {
            session.owners.emplace_back(std::move(private_owner));
        } catch (...) {
            program.continuation_slots[slot_index].role = ContinuationSlotRole::Free;
            throw;
        }
    } else {
        auto& shared_owner = std::get<ContextCacheImportedSharedOwner>(owner);
        shared_owner.slot_index = reserve_shared_owner_slot(program, session);
        const std::uint32_t slot_index = shared_owner.slot_index;
        try {
            session.owners.emplace_back(std::move(shared_owner));
        } catch (...) {
            program.shared_prefix_slots[slot_index].role = SharedPrefixSlotRole::Free;
            throw;
        }
    }
}

} // namespace

std::shared_ptr<void> ProgramImplCore::begin_context_cache_export() {
    if (has_context_transaction() || pending_transaction_ || pressure_planning_active_ ||
        causal_scoring) {
        throw std::logic_error("context-cache export requires an idle generation Program");
    }
    require_snapshot_profile(*this);
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        const ContinuationSlotRole role = continuation_slots[index].role;
        if (role != ContinuationSlotRole::Free && role != ContinuationSlotRole::Catalogued) {
            throw std::logic_error("context-cache export found a nonterminal private owner");
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        const SharedPrefixSlotRole role = shared_prefix_slots[index].role;
        if (role != SharedPrefixSlotRole::Free && role != SharedPrefixSlotRole::Catalogued) {
            throw std::logic_error("context-cache export found a claimed shared owner");
        }
    }
    auto state = std::make_shared<ContextCacheExportState>();
    state->owner = this;
    return state;
}

void ProgramImplCore::write_context_cache_owner(const std::shared_ptr<void>& opaque,
                                               const ContinuationHandle& owner,
                                               ContextCacheWriter& writer) const {
    ContextCacheExportState& session = require_export_session(*this, opaque);
    if (!valid_continuation(owner)) {
        throw ninfer::ContextCacheOwnershipError("context-cache private capability is stale");
    }
    const std::uint32_t index = ContractAccess::index(owner);
    const std::uint64_t generation = ContractAccess::epoch(owner);
    if (continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[index].generation != generation) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache private owner is not an inactive catalogue entry");
    }
    const OwnerCapabilityKey owner_key = owner_link_key(index, generation);
    if (!session.private_owners.insert(owner_key).second) {
        throw std::invalid_argument("context-cache private owner was exported twice");
    }
    try {
        ContextCacheExportState sizing_session = session;
        CountingContextCacheWriter sizing_writer;
        write_private_owner_record(*const_cast<ProgramImplCore*>(this), sizing_session,
                                   continuation_states[index], sizing_writer);
        write_u32(writer, kOwnerRecordMagic);
        write_u32(writer, kOwnerRecordVersion);
        write_u64(writer, sizing_writer.bytes());
        write_private_owner_record(*const_cast<ProgramImplCore*>(this), session,
                                   continuation_states[index], writer);
    } catch (...) {
        session.closed = true;
        throw;
    }
}

void ProgramImplCore::write_context_cache_owner(const std::shared_ptr<void>& opaque,
                                                const SharedPrefixHandle& owner,
                                                ContextCacheWriter& writer) const {
    ContextCacheExportState& session = require_export_session(*this, opaque);
    if (!valid_shared_prefix(owner)) {
        throw ninfer::ContextCacheOwnershipError("context-cache shared capability is stale");
    }
    const std::uint32_t index = ContractAccess::index(owner);
    const std::uint64_t generation = ContractAccess::epoch(owner);
    if (shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
        shared_prefix_slots[index].generation != generation ||
        shared_prefix_states[index].active_references != 0) {
        throw ninfer::ContextCacheOwnershipError(
            "context-cache shared owner is not an inactive catalogue entry");
    }
    const OwnerCapabilityKey owner_key = owner_link_key(index, generation);
    if (!session.shared_owners.insert(owner_key).second) {
        throw std::invalid_argument("context-cache shared owner was exported twice");
    }
    try {
        ContextCacheExportState sizing_session = session;
        CountingContextCacheWriter sizing_writer;
        write_shared_owner_record(*const_cast<ProgramImplCore*>(this), sizing_session,
                                  shared_prefix_states[index], sizing_writer);
        write_u32(writer, kOwnerRecordMagic);
        write_u32(writer, kOwnerRecordVersion);
        write_u64(writer, sizing_writer.bytes());
        write_shared_owner_record(*const_cast<ProgramImplCore*>(this), session,
                                  shared_prefix_states[index], writer);
    } catch (...) {
        session.closed = true;
        throw;
    }
}

void ProgramImplCore::end_context_cache_export(std::shared_ptr<void>&& opaque) noexcept {
    if (!opaque) { return; }
    auto* session = static_cast<ContextCacheExportState*>(opaque.get());
    if (session->owner == this) { session->closed = true; }
    opaque.reset();
}

std::shared_ptr<void> ProgramImplCore::begin_context_cache_import(
    std::size_t expected_owner_count) {
    if (has_context_transaction() || pending_transaction_ || pressure_planning_active_ ||
        causal_scoring) {
        throw std::logic_error("context-cache import requires an idle generation Program");
    }
    require_snapshot_profile(*this);
    if (expected_owner_count > static_cast<std::size_t>(continuation_capacity) +
                                   static_cast<std::size_t>(shared_prefix_capacity)) {
        throw std::invalid_argument("context-cache owner count exceeds Program capacity");
    }
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty) {
            throw std::logic_error("context-cache import cannot overlap a live request");
        }
    }
    for (const ContinuationSlot& slot : continuation_slots) {
        if (slot.role != ContinuationSlotRole::Free) {
            throw std::logic_error("context-cache import requires empty private owner slots");
        }
    }
    for (const SharedPrefixSlot& slot : shared_prefix_slots) {
        if (slot.role != SharedPrefixSlotRole::Free) {
            throw std::logic_error("context-cache import requires empty shared owner slots");
        }
    }

    auto state = std::make_shared<ContextCacheImportState>();
    state->owner = this;
    state->expected_owner_count = expected_owner_count;
    state->owners.reserve(expected_owner_count);
    state->returned_owners.reserve(expected_owner_count);
    state->private_slot_indices.reserve(expected_owner_count);
    state->shared_slot_indices.reserve(expected_owner_count);
    if (expected_owner_count > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("context-cache owner address staging capacity overflows");
    }
    state->allocated_addresses.reserve(expected_owner_count * 2U);
    if (host_kv_extents) {
        state->published_host_extents.reserve(host_kv_extents->capacity());
    }
    if (expected_owner_count <= std::numeric_limits<std::size_t>::max() / 2U) {
        state->object_indices.reserve(expected_owner_count * 2U);
    }
    state->main_reservation = text_kv_pages->physical_pool().make_empty_reservation();
    if (backend_kv_pages) {
        state->backend_reservation = backend_kv_pages->physical_pool().make_empty_reservation();
    }
    return state;
}

qwen3_6::ContextCacheOwnerSummary ProgramImplCore::read_context_cache_owner(
    const std::shared_ptr<void>& opaque, ContextCacheReader& reader,
    ContextCacheOwnerKind expected) {
    ContextCacheImportState& session = require_import_session(*this, opaque);
    if (session.owners.size() >= session.expected_owner_count) {
        throw std::invalid_argument("context-cache archive has more owners than declared");
    }
    if (expected != ContextCacheOwnerKind::PrivateContinuation &&
        expected != ContextCacheOwnerKind::SharedPrefix) {
        throw std::invalid_argument("context-cache owner kind is unknown");
    }
    const std::uint32_t magic = read_u32(reader);
    const std::uint32_t version = read_u32(reader);
    const std::uint64_t body_bytes = read_u64(reader);
    if (magic != kOwnerRecordMagic || version != kOwnerRecordVersion || body_bytes == 0 ||
        body_bytes > reader.remaining_bytes()) {
        throw std::invalid_argument("context-cache owner record framing is invalid");
    }
    BoundedContextCacheReader bounded(reader, body_bytes);
    validate_target_header(bounded, *this, expected);
    qwen3_6::ContextCacheOwnerSummary summary;
    if (expected == ContextCacheOwnerKind::PrivateContinuation) {
        ContextCacheImportedOwner owner = read_private_owner_body(*this, session, bounded);
        summary = std::get<ContextCacheImportedPrivateOwner>(owner).summary;
        register_import_owner(*this, session, std::move(owner));
    } else {
        ContextCacheImportedOwner owner = read_shared_owner_body(*this, session, bounded);
        summary = std::get<ContextCacheImportedSharedOwner>(owner).summary;
        register_import_owner(*this, session, std::move(owner));
    }
    if (bounded.remaining_bytes() != 0) {
        throw std::invalid_argument("context-cache owner record contains trailing bytes");
    }
    return summary;
}

#include "targets/qwen3_6/impl/runtime/context_cache_snapshot_commit_methods.h"
