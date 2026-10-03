#pragma once

std::vector<qwen3_6::ContextCacheOwnerImport<Variant>>
ProgramImplCore::commit_context_cache_import(std::shared_ptr<void>& opaque) {
    static_assert(std::is_nothrow_move_assignable_v<SequenceState>);
    static_assert(std::is_nothrow_move_assignable_v<SharedPrefixState>);
    static_assert(std::is_nothrow_move_constructible_v<
                  std::vector<qwen3_6::ContextCacheOwnerImport<Variant>>>);
    if (!opaque) { throw std::invalid_argument("context-cache import session is empty"); }
    ContextCacheImportState& session = require_import_session(*this, opaque);

    try {
        if (has_context_transaction() || pending_transaction_ || pressure_planning_active_ ||
            causal_scoring || !state_store || !text_kv_pages || !text_kv_addresses) {
            throw std::logic_error("context-cache import commit requires an idle generation Program");
        }
        if (session.owners.size() != session.expected_owner_count ||
            session.returned_owners.size() != 0 ||
            session.returned_owners.capacity() < session.expected_owner_count ||
            !session.published_host_extents.empty()) {
            throw std::invalid_argument("context-cache import staging is incomplete");
        }

        struct AddressInstallPlan {
            ContextCacheImportAddress* staged = nullptr;
            KVAddressSpaceStore* store = nullptr;
            std::vector<LogicalKVPageHandle> pages;
        };

        std::size_t maximum_addresses = 0;
        if (session.owners.size() > std::numeric_limits<std::size_t>::max() / 2U) {
            throw std::overflow_error("context-cache imported address count overflows");
        }
        maximum_addresses = session.owners.size() * 2U;
        std::vector<AddressInstallPlan> address_plans;
        address_plans.reserve(maximum_addresses);
        std::vector<std::uint32_t> observed_page_references(session.objects.size(), 0);
        std::vector<std::uint8_t> private_slots_seen(continuation_capacity, 0);
        std::vector<std::uint8_t> shared_slots_seen(shared_prefix_capacity, 0);

        std::size_t private_owner_count = 0;
        std::size_t shared_owner_count = 0;
        std::size_t host_page_count = 0;
        bool needs_main_device_reservation = false;
        bool needs_backend_device_reservation = false;

        const auto add_address = [&](ContextCacheImportAddress& address, bool expected_backend,
                                     std::uint32_t expected_frontier) {
            if (address.backend != expected_backend || address.frontier != expected_frontier ||
                !address.address.valid() || address.installed) {
                throw std::invalid_argument("context-cache imported address metadata is inconsistent");
            }
            KVAddressSpaceStore* store = expected_backend ? backend_kv_addresses.get()
                                                           : text_kv_addresses.get();
            LogicalKVPageStore* pages = expected_backend ? backend_kv_pages.get()
                                                          : text_kv_pages.get();
            const SnapshotObjectKind kind = expected_backend ? SnapshotObjectKind::BackendKVPage
                                                              : SnapshotObjectKind::MainKVPage;
            if (store == nullptr || pages == nullptr || !store->valid(address.address) ||
                store->active(address.address) || store->bound_row(address.address) >= 0 ||
                store->mapped_pages(address.address) != 0 ||
                store->committed_frontier(address.address) != 0 ||
                address.page_links.size() != page_count_for_frontier(expected_frontier)) {
                throw std::invalid_argument("context-cache imported address store or shape is invalid");
            }
            address_plans.emplace_back();
            AddressInstallPlan& plan = address_plans.back();
            plan.staged = &address;
            plan.store = store;
            plan.pages.reserve(address.page_links.size());
            for (const std::uint64_t link : address.page_links) {
                const std::size_t object_index = lookup_import_link(session, link, kind);
                const ContextCacheImportObject& object = session.objects[object_index];
                if (!object.page.valid() || !pages->valid(object.page) ||
                    observed_page_references[object_index] ==
                        std::numeric_limits<std::uint32_t>::max()) {
                    throw std::invalid_argument("context-cache imported page link is invalid");
                }
                const std::uint32_t page_begin =
                    static_cast<std::uint32_t>(plan.pages.size()) *
                    static_cast<std::uint32_t>(kPagedKVPageSize);
                const std::uint32_t needed = std::min(
                    static_cast<std::uint32_t>(kPagedKVPageSize), expected_frontier - page_begin);
                if (pages->committed_columns(object.page) < needed ||
                    pages->writer_references(object.page) != 0 ||
                    pages->source_pins(object.page) != 0 ||
                    pages->active_address_references(object.page) != 0 ||
                    (!object.device_replica && !object.host_replica)) {
                    throw std::invalid_argument("context-cache imported page coverage is incomplete");
                }
                if (std::find(plan.pages.begin(), plan.pages.end(), object.page) != plan.pages.end()) {
                    throw std::invalid_argument("context-cache imported address repeats a page");
                }
                ++observed_page_references[object_index];
                plan.pages.push_back(object.page);
            }
        };

        for (ContextCacheImportedOwner& imported : session.owners) {
            if (auto* owner = std::get_if<ContextCacheImportedPrivateOwner>(&imported)) {
                ++private_owner_count;
                if (owner->slot_index >= continuation_capacity ||
                    continuation_slots[owner->slot_index].role !=
                        ContinuationSlotRole::ReservedMaterialization ||
                    private_slots_seen[owner->slot_index] != 0 ||
                    owner->summary.active_references != 0 ||
                    owner->sequence.endpoint_valid != owner->sequence.state.read.valid() ||
                    owner->sequence.endpoint_valid != owner->sequence.state.write.valid() ||
                    (owner->sequence.endpoint_valid &&
                     owner->sequence.state.read != owner->sequence.state.write) ||
                    (!owner->sequence.endpoint_valid &&
                     (owner->sequence.state.read.valid() || owner->sequence.state.write.valid())) ||
                    owner->sequence.shared_prefix_references.size() != 0) {
                    throw std::invalid_argument("context-cache private owner staging is inconsistent");
                }
                private_slots_seen[owner->slot_index] = 1;
                const bool expects_backend = speculative_backend == SpeculativeBackend::Mtp;
                if (expects_backend != owner->backend_address.has_value() ||
                    owner->sequence.text_kv_valid != owner->main_address.frontier ||
                    owner->main_address.frontier > owner->sequence.execution_frontier ||
                    (owner->sequence.endpoint_valid &&
                     owner->sequence.execution_frontier != owner->main_address.frontier) ||
                    (expects_backend &&
                     owner->sequence.mtp_kv_valid != owner->backend_address->frontier) ||
                    (!expects_backend && owner->sequence.mtp_kv_valid != 0)) {
                    throw std::invalid_argument("context-cache private KV frontiers are inconsistent");
                }
                owner->sequence.kv = SequenceKVBundle{
                    .text = owner->main_address.address,
                    .backend = owner->backend_address
                                   ? std::optional<KVAddressSpaceHandle>(
                                         owner->backend_address->address)
                                   : std::nullopt,
                };
                const qwen3_6::ContinuationSummary validated =
                    make_imported_continuation_summary(*this, owner->sequence);
                if (validated != owner->summary) {
                    throw std::invalid_argument("context-cache private owner summary changed before commit");
                }
                const SnapshotRequiredFrontiers required =
                    private_required_frontiers(owner->summary);
                if (owner->main_address.checkpoint_frontier < required.main ||
                    (owner->backend_address &&
                     owner->backend_address->checkpoint_frontier < required.backend)) {
                    throw std::invalid_argument(
                        "context-cache private KV protection is below its checkpoint requirement");
                }
                refresh_state_views(owner->sequence);
                add_address(owner->main_address, false, owner->sequence.text_kv_valid);
                if (owner->backend_address) {
                    add_address(*owner->backend_address, true, owner->sequence.mtp_kv_valid);
                }
            } else {
                auto& shared_owner = std::get<ContextCacheImportedSharedOwner>(imported);
                ++shared_owner_count;
                if (shared_owner.slot_index >= shared_prefix_capacity ||
                    shared_prefix_slots[shared_owner.slot_index].role !=
                        SharedPrefixSlotRole::ReservedCapture ||
                    shared_slots_seen[shared_owner.slot_index] != 0 ||
                    shared_owner.summary.active_references != 0 ||
                    shared_owner.shared.active_references != 0 ||
                    shared_owner.shared.frontier != shared_owner.main_address.frontier ||
                    (speculative_backend == SpeculativeBackend::Mtp) !=
                        shared_owner.backend_address.has_value() ||
                    (shared_owner.backend_address &&
                     (shared_owner.shared.backend_frontier >
                          shared_owner.backend_address->frontier ||
                      shared_owner.backend_address->frontier > shared_owner.shared.frontier ||
                      shared_owner.backend_address->checkpoint_frontier <
                          shared_owner.shared.backend_frontier)) ||
                    shared_owner.main_address.checkpoint_frontier <
                        shared_owner.shared.frontier ||
                    (!shared_owner.backend_address &&
                     shared_owner.shared.backend_frontier != 0)) {
                    throw std::invalid_argument("context-cache shared owner staging is inconsistent");
                }
                shared_slots_seen[shared_owner.slot_index] = 1;
                shared_owner.shared.kv = SequenceKVBundle{
                    .text = shared_owner.main_address.address,
                    .backend = shared_owner.backend_address
                                   ? std::optional<KVAddressSpaceHandle>(
                                         shared_owner.backend_address->address)
                                   : std::nullopt,
                };
                const qwen3_6::SharedPrefixSummary validated =
                    make_imported_shared_summary(*this, shared_owner.shared);
                if (validated != shared_owner.summary) {
                    throw std::invalid_argument("context-cache shared owner summary changed before commit");
                }
                add_address(shared_owner.main_address, false, shared_owner.shared.frontier);
                if (shared_owner.backend_address) {
                    add_address(*shared_owner.backend_address, true,
                                shared_owner.backend_address->frontier);
                }
            }
        }

        if (private_owner_count != session.private_slot_indices.size() ||
            shared_owner_count != session.shared_slot_indices.size() ||
            address_plans.size() != session.allocated_addresses.size()) {
            throw std::invalid_argument("context-cache imported owner reservations are incomplete");
        }
        for (const ContextCacheStagedAddress& allocated : session.allocated_addresses) {
            if (!allocated.address.valid()) {
                throw std::invalid_argument("context-cache imported address reservation is empty");
            }
            std::size_t matches = 0;
            for (const AddressInstallPlan& plan : address_plans) {
                if (plan.staged->backend == allocated.backend &&
                    plan.staged->address == allocated.address) {
                    ++matches;
                }
            }
            if (matches != 1) {
                throw std::invalid_argument("context-cache imported address reservation is duplicated");
            }
        }
        for (const std::uint32_t index : session.private_slot_indices) {
            if (index >= continuation_capacity || private_slots_seen[index] != 1) {
                throw std::invalid_argument("context-cache private slot reservation is unowned");
            }
            private_slots_seen[index] = 2;
        }
        for (const std::uint32_t index : session.shared_slot_indices) {
            if (index >= shared_prefix_capacity || shared_slots_seen[index] != 1) {
                throw std::invalid_argument("context-cache shared slot reservation is unowned");
            }
            shared_slots_seen[index] = 2;
        }

        for (std::size_t index = 0; index < session.objects.size(); ++index) {
            ContextCacheImportObject& object = session.objects[index];
            if (object.link_id == 0 || object.published || object.published_host_extent ||
                session.object_indices.find(object.link_id) == session.object_indices.end() ||
                session.object_indices.at(object.link_id) != index) {
                throw std::invalid_argument("context-cache imported object map is inconsistent");
            }
            if (object.kind == SnapshotObjectKind::StateImage) {
                if (!object.state.valid() || !state_store->valid(object.state) || object.page.valid() ||
                    (object.state_references == 0 && object.primary_owners == 0) ||
                    object.primary_owners > 1 || object.address_references != 0 ||
                    state_store->role(object.state) != StateImageRole::ReservedDestination ||
                    state_store->checkpoint_references(object.state) != 0 ||
                    (object.device_replica !=
                     (state_store->residency(object.state) == StateReplicaResidency::DeviceOnly ||
                      state_store->residency(object.state) == StateReplicaResidency::Both)) ||
                    (object.host_replica !=
                     (state_store->residency(object.state) == StateReplicaResidency::HostOnly ||
                      state_store->residency(object.state) == StateReplicaResidency::Both)) ||
                    object.host_allocation) {
                    throw std::invalid_argument("context-cache imported StateImage is not publishable");
                }
            } else if (object.kind == SnapshotObjectKind::MainKVPage ||
                       object.kind == SnapshotObjectKind::BackendKVPage) {
                const bool backend = object.kind == SnapshotObjectKind::BackendKVPage;
                LogicalKVPageStore* pages = backend ? backend_kv_pages.get() : text_kv_pages.get();
                if (pages == nullptr || !object.page.valid() || !pages->valid(object.page) ||
                    object.state.valid() || object.state_references != 0 ||
                    object.primary_owners != 0 || object.address_references == 0 ||
                    observed_page_references[index] != object.address_references ||
                    object.committed_columns == 0 ||
                    object.committed_columns > pages->physical_pool().geometry().page_tokens ||
                    pages->committed_columns(object.page) != object.committed_columns ||
                    pages->address_references(object.page) != 0 ||
                    pages->writer_references(object.page) != 0 ||
                    pages->active_address_references(object.page) != 0 ||
                    pages->source_pins(object.page) != 0 ||
                    pages->device_resident(object.page) != object.device_replica ||
                    pages->host_resident(object.page) ||
                    object.host_replica != object.host_allocation.has_value() ||
                    (!object.device_replica && !object.host_replica)) {
                    throw std::invalid_argument("context-cache imported KV page is not publishable");
                }
                if (object.host_allocation) {
                    if (!host_kv_arena || !host_kv_extents ||
                        !object.host_allocation->valid() || object.host_allocation->page_count() != 1) {
                        throw std::invalid_argument("context-cache imported Host KV allocation is invalid");
                    }
                    const HostKVAllocationConstView view =
                        host_kv_arena->view(*object.host_allocation);
                    const HostKVPageLayout layout =
                        plan_host_kv_page_layout(pages->physical_pool().geometry());
                    if (view.layout().geometry != pages->physical_pool().geometry() ||
                        view.layout().page_stride != layout.page_stride ||
                        view.page_count() != 1) {
                        throw std::invalid_argument("context-cache imported Host KV geometry changed");
                    }
                    ++host_page_count;
                }
                if (object.device_replica) {
                    if (backend) {
                        needs_backend_device_reservation = true;
                    } else {
                        needs_main_device_reservation = true;
                    }
                }
            } else {
                throw std::invalid_argument("context-cache imported object kind is unknown");
            }
        }

        if (session.object_indices.size() != session.objects.size() ||
            (host_page_count != 0 &&
             (host_kv_extents == nullptr ||
              !host_kv_extents->can_publish_imported_pages(host_page_count) ||
              session.published_host_extents.capacity() < host_page_count)) ||
            (needs_main_device_reservation &&
             (!session.main_reservation.valid() ||
              !session.main_reservation.belongs_to(text_kv_pages->physical_pool()))) ||
            (needs_backend_device_reservation &&
             (!backend_kv_pages || !session.backend_reservation.valid() ||
              !session.backend_reservation.belongs_to(backend_kv_pages->physical_pool())))) {
            throw std::invalid_argument("context-cache import physical capacity or reservation changed");
        }

        // Build the complete public result before publishing any physical owner. This performs
        // any summary/vector copies while rollback is still an ordinary unpublished import.
        for (ContextCacheImportedOwner& imported : session.owners) {
            if (auto* owner = std::get_if<ContextCacheImportedPrivateOwner>(&imported)) {
                qwen3_6::ImportedPrivateContextCacheOwner<Variant> result{
                    .handle = ContractAccess::make_continuation(
                        this, owner->slot_index, continuation_slots[owner->slot_index].generation),
                    .summary = owner->summary,
                };
                session.returned_owners.emplace_back(std::move(result));
            } else {
                auto& shared_owner = std::get<ContextCacheImportedSharedOwner>(imported);
                qwen3_6::ImportedSharedContextCacheOwner<Variant> result{
                    .handle = ContractAccess::make_shared_prefix(
                        this, shared_owner.slot_index,
                        shared_prefix_slots[shared_owner.slot_index].generation),
                    .summary = shared_owner.summary,
                };
                session.returned_owners.emplace_back(std::move(result));
            }
        }
        std::vector<qwen3_6::ContextCacheOwnerImport<Variant>> returned =
            std::move(session.returned_owners);

        // Host-only pages are not resident from the page store's point of view until their
        // extent is attached, so publish the already validated Host allocations before installing
        // address membership. The extent ledger makes this first publication rollbackable.
        for (ContextCacheImportObject& object : session.objects) {
            if (!object.host_allocation) { continue; }
            LogicalKVPageStore& pages = object.kind == SnapshotObjectKind::BackendKVPage
                                            ? *backend_kv_pages
                                            : *text_kv_pages;
            HostKVExtentCapability extent = host_kv_extents->publish_imported_page(
                pages, object.page, std::move(*object.host_allocation));
            object.host_allocation.reset();
            object.published_host_extent = extent;
            session.published_host_extents.push_back(extent);
        }

        for (AddressInstallPlan& plan : address_plans) {
            plan.store->install_imported(
                plan.staged->address, plan.staged->frontier,
                std::span<const LogicalKVPageHandle>(plan.pages));
            plan.staged->installed = true;
            plan.store->set_checkpoint_requirement(plan.staged->address,
                                                    plan.staged->checkpoint_frontier);
        }

        // All remaining operations are fixed-capacity/noexcept publications. The complete
        // reference totals were validated above, so no owner is visible with partial coverage.
        for (ContextCacheImportObject& object : session.objects) {
            if (object.kind == SnapshotObjectKind::StateImage) {
                state_store->publish_imported_checkpoint(
                    object.state, object.state_references, object.primary_owners != 0);
            } else {
                LogicalKVPageStore& pages = object.kind == SnapshotObjectKind::BackendKVPage
                                                ? *backend_kv_pages
                                                : *text_kv_pages;
                pages.publish_imported_page(object.page, object.address_references);
            }
            object.published = true;
        }

        for (ContextCacheImportedOwner& imported : session.owners) {
            if (auto* owner = std::get_if<ContextCacheImportedPrivateOwner>(&imported)) {
                continuation_states[owner->slot_index] = std::move(owner->sequence);
                continuation_slots[owner->slot_index].role = ContinuationSlotRole::Catalogued;
            } else {
                auto& shared_owner = std::get<ContextCacheImportedSharedOwner>(imported);
                shared_prefix_states[shared_owner.slot_index] =
                    std::move(shared_owner.shared);
                shared_prefix_slots[shared_owner.slot_index].role =
                    SharedPrefixSlotRole::Catalogued;
            }
        }

        session.main_reservation.release();
        session.backend_reservation.release();
        if (session.expected_owner_count != 0) { advance_resource_revision(); }
        session.committed = true;
        return returned;
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (!abort_context_cache_import(std::move(opaque))) {
            throw ninfer::ContextCacheOwnershipError(
                "context-cache import publication could not be rolled back");
        }
        std::rethrow_exception(failure);
    }
}

bool ProgramImplCore::abort_context_cache_import(std::shared_ptr<void>&& opaque) noexcept {
    if (!opaque) { return true; }
    auto* session = static_cast<ContextCacheImportState*>(opaque.get());
    if (session->owner != this) { return false; }
    if (session->committed) {
        opaque.reset();
        return true;
    }

    bool rollback_ok = true;
    const auto address_store = [&](bool backend) noexcept -> KVAddressSpaceStore* {
        return backend ? backend_kv_addresses.get() : text_kv_addresses.get();
    };

    // Roll back all address memberships before detaching imported Host extents or page objects.
    for (ContextCacheStagedAddress& staged : session->allocated_addresses) {
        if (!staged.address.valid()) { continue; }
        KVAddressSpaceStore* store = address_store(staged.backend);
        if (store == nullptr || !store->valid(staged.address) ||
            !store->rollback_imported(staged.address)) {
            rollback_ok = false;
            continue;
        }
        if (!store->release(staged.address)) {
            rollback_ok = false;
            continue;
        }
        staged.address = KVAddressSpaceHandle{};
    }
    for (ContextCacheImportedOwner& imported : session->owners) {
        if (auto* owner = std::get_if<ContextCacheImportedPrivateOwner>(&imported)) {
            owner->main_address.address = {};
            owner->main_address.installed = false;
            if (owner->backend_address) {
                owner->backend_address->address = {};
                owner->backend_address->installed = false;
            }
        } else {
            auto& shared_owner = std::get<ContextCacheImportedSharedOwner>(imported);
            shared_owner.main_address.address = {};
            shared_owner.main_address.installed = false;
            if (shared_owner.backend_address) {
                shared_owner.backend_address->address = {};
                shared_owner.backend_address->installed = false;
            }
        }
    }

    if (host_kv_extents != nullptr) {
        for (HostKVExtentCapability& extent : session->published_host_extents) {
            if (!extent.valid() || !host_kv_extents->valid(extent)) { continue; }
            if (!host_kv_extents->release(extent)) {
                rollback_ok = false;
            } else {
                extent = HostKVExtentCapability{};
            }
        }
    } else if (!session->published_host_extents.empty()) {
        rollback_ok = false;
    }

    for (ContextCacheImportObject& object : session->objects) {
        if (object.kind == SnapshotObjectKind::StateImage) {
            if (!object.state.valid()) { continue; }
            if (state_store == nullptr || !state_store->valid(object.state) ||
                !state_store->release(object.state)) {
                rollback_ok = false;
                continue;
            }
            object.state = StateImageHandle{};
            object.published_host_extent.reset();
            object.host_allocation.reset();
            continue;
        }

        LogicalKVPageStore* pages = object.kind == SnapshotObjectKind::BackendKVPage
                                        ? backend_kv_pages.get()
                                        : object.kind == SnapshotObjectKind::MainKVPage
                                              ? text_kv_pages.get()
                                              : nullptr;
        if (pages == nullptr || !object.page.valid() || !pages->valid(object.page)) {
            if (object.page.valid()) { rollback_ok = false; }
            continue;
        }
        DeviceKVPageReservation& reservation = object.kind == SnapshotObjectKind::BackendKVPage
                                                   ? session->backend_reservation
                                                   : session->main_reservation;
        pages->abort_import_destination(object.page, reservation);
        if (pages->valid(object.page)) {
            rollback_ok = false;
            continue;
        }
        object.page = LogicalKVPageHandle{};
        object.published_host_extent.reset();
        object.host_allocation.reset();
    }

    if (rollback_ok) {
        for (const std::uint32_t index : session->private_slot_indices) {
            if (index >= continuation_capacity) { rollback_ok = false; continue; }
            ContinuationSlot& slot = continuation_slots[index];
            if (slot.role == ContinuationSlotRole::Free) { continue; }
            if (slot.role != ContinuationSlotRole::ReservedMaterialization) {
                rollback_ok = false;
                continue;
            }
            continuation_states[index] = SequenceState{};
            slot.role = ContinuationSlotRole::Free;
            if (++slot.generation == 0) { ++slot.generation; }
        }
        for (const std::uint32_t index : session->shared_slot_indices) {
            if (index >= shared_prefix_capacity) { rollback_ok = false; continue; }
            SharedPrefixSlot& slot = shared_prefix_slots[index];
            if (slot.role == SharedPrefixSlotRole::Free) { continue; }
            if (slot.role != SharedPrefixSlotRole::ReservedCapture) {
                rollback_ok = false;
                continue;
            }
            shared_prefix_states[index] = SharedPrefixState{};
            slot.role = SharedPrefixSlotRole::Free;
            if (++slot.generation == 0) { ++slot.generation; }
        }
    }

    if (!rollback_ok) {
        session->rollback_failed = true;
        return false;
    }
    session->main_reservation.release();
    session->backend_reservation.release();
    opaque.reset();
    return true;
}
