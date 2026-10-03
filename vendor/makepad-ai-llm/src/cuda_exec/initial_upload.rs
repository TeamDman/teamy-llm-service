//! Windows initial arena upload through two bounded CUDA-pinned slots.
//!
//! A slot's completion event is waited before its host bytes are overwritten.
//! All copies use the runtime's existing stream and original logical offsets;
//! no tensor representation, kernel or graph is changed. The local owner drains
//! pending DMA before freeing pinned memory, including when enqueueing or
//! recording the completion event fails.

use std::ffi::c_void;
use std::time::Instant;

use super::{
    check, cudaDeviceSynchronize, cudaEventCreate, cudaEventDestroy, cudaEventElapsedTime,
    cudaEventRecord, cudaFreeHost, cudaHostAlloc, cudaMemcpyAsync, cudaMemsetAsync,
    cudaStreamSynchronize, cuda_err_str, Arena, Context, CudaErr, CudaEvent, LlamaError, Result,
    CUDA_MEMCPY_HOST_TO_DEVICE, CUDA_SUCCESS,
};

const SLOT_COUNT: usize = 2;
const MAX_SLOT_BYTES: usize = 64 << 20;
const WRITE_COMBINED_ENV: &str = "MAKEPAD_LLAMA_CUDA_UPLOAD_WRITE_COMBINED";
const COPY_THREADS_ENV: &str = "MAKEPAD_LLAMA_CUDA_UPLOAD_COPY_THREADS";
const PARALLEL_COPY_MIN_BYTES: usize = 1 << 20;
// cudaHostAllocWriteCombined, driver_types.h. This remains opt-in until
// same-build measurements establish whether it helps this producer-only path.
const CUDA_HOST_ALLOC_WRITE_COMBINED: u32 = 4;

fn experimental_host_alloc_flags(value: Option<&str>) -> Result<u32> {
    match value {
        None | Some("0") => Ok(0),
        Some("1") => Ok(CUDA_HOST_ALLOC_WRITE_COMBINED),
        Some(_) => Err(LlamaError::format(format!(
            "{WRITE_COMBINED_ENV} must be 0 or 1 for the initial-upload experiment"
        ))),
    }
}

fn process_host_alloc_flags() -> Result<u32> {
    let value = std::env::var_os(WRITE_COMBINED_ENV);
    let value = value
        .as_ref()
        .map(|value| {
            value.to_str().ok_or_else(|| {
                LlamaError::format(format!("{WRITE_COMBINED_ENV} must be valid UTF-8"))
            })
        })
        .transpose()?;
    experimental_host_alloc_flags(value)
}

fn experimental_copy_threads(value: Option<&str>) -> Result<usize> {
    match value {
        None | Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some("4") => Ok(4),
        Some(_) => Err(LlamaError::format(format!(
            "{COPY_THREADS_ENV} must be 1, 2 or 4 for the initial-upload experiment"
        ))),
    }
}

fn process_copy_threads() -> Result<usize> {
    let value = std::env::var_os(COPY_THREADS_ENV);
    let value = value
        .as_ref()
        .map(|value| {
            value.to_str().ok_or_else(|| {
                LlamaError::format(format!("{COPY_THREADS_ENV} must be valid UTF-8"))
            })
        })
        .transpose()?;
    experimental_copy_threads(value)
}

fn copy_staging_bytes(source: &[u8], destination: &mut [u8], threads: usize) -> Result<()> {
    copy_staging_bytes_with(
        source,
        destination,
        threads,
        &|_| Ok(()),
        &|_, source, destination| destination.copy_from_slice(source),
    )
}

/// Hooks allow model-free failures without exhausting OS thread resources.
/// Production uses a no-op pre-spawn hook and an exact byte copy. Every started
/// worker is joined, even after a later spawn failure or another worker panic.
fn copy_staging_bytes_with(
    source: &[u8],
    destination: &mut [u8],
    threads: usize,
    before_spawn: &impl Fn(usize) -> std::io::Result<()>,
    copy: &(impl Fn(usize, &[u8], &mut [u8]) + Sync),
) -> Result<()> {
    if source.len() != destination.len() {
        return Err(LlamaError::format(
            "pinned staging source/destination lengths differ",
        ));
    }
    if !matches!(threads, 1 | 2 | 4) {
        return Err(LlamaError::format(
            "pinned staging copy thread count must be 1, 2 or 4",
        ));
    }
    if threads == 1 || source.len() < PARALLEL_COPY_MIN_BYTES {
        copy(0, source, destination);
        return Ok(());
    }

    let chunk_bytes = source.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(threads);
        let mut first_error = None;
        for (index, (source, destination)) in source
            .chunks(chunk_bytes)
            .zip(destination.chunks_mut(chunk_bytes))
            .enumerate()
        {
            let spawned = before_spawn(index).and_then(|()| {
                std::thread::Builder::new()
                    .name(format!("cuda-upload-copy-{index}"))
                    .spawn_scoped(scope, move || copy(index, source, destination))
            });
            match spawned {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    first_error = Some(LlamaError::format(format!(
                        "spawning pinned staging copy worker {index} failed: {error}"
                    )));
                    break;
                }
            }
        }
        for (index, worker) in workers.into_iter().enumerate() {
            if worker.join().is_err() && first_error.is_none() {
                first_error = Some(LlamaError::format(format!(
                    "pinned staging copy worker {index} panicked"
                )));
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

extern "C" {
    fn cudaEventSynchronize(event: CudaEvent) -> CudaErr;
}

/// The scheduler can be exercised without a CUDA device. `wait` must establish
/// that the last copy from this slot has finished before `stage` reuses it.
trait UploadTransport {
    fn capacity(&self) -> usize;
    fn wait(&mut self, slot: usize) -> Result<()>;
    fn stage(&mut self, slot: usize, source: &[u8]) -> Result<()>;
    fn enqueue(&mut self, slot: usize, destination: usize, len: usize) -> Result<()>;
}

fn upload_region<'a>(
    transport: &mut impl UploadTransport,
    source_start: usize,
    len: usize,
    mut source: impl FnMut(usize, usize) -> Result<&'a [u8]>,
    progress: &mut dyn FnMut(usize),
) -> Result<()> {
    let capacity = transport.capacity();
    if capacity == 0 {
        return Err(LlamaError::format("pinned upload slot capacity is zero"));
    }
    source_start
        .checked_add(len)
        .ok_or_else(|| LlamaError::format("pinned upload source range overflows usize"))?;

    let mut pending = [None; SLOT_COUNT];
    let mut offset = 0usize;
    let mut next_slot = 0usize;
    while offset < len {
        if let Some(completed) = pending[next_slot].take() {
            transport.wait(next_slot)?;
            progress(completed);
        }
        let chunk_len = capacity.min(len - offset);
        let absolute = source_start + offset;
        let bytes = source(absolute, chunk_len)?;
        if bytes.len() != chunk_len {
            return Err(LlamaError::format(
                "pinned upload source returned the wrong length",
            ));
        }
        transport.stage(next_slot, bytes)?;
        transport.enqueue(next_slot, absolute, chunk_len)?;
        offset += chunk_len;
        pending[next_slot] = Some(offset);
        next_slot = (next_slot + 1) % SLOT_COUNT;
    }

    // The next reuse slot is the oldest pending copy. Waiting in submission
    // order preserves monotone *completed* progress even for an odd chunk count.
    for step in 0..SLOT_COUNT {
        let slot = (next_slot + step) % SLOT_COUNT;
        if let Some(completed) = pending[slot].take() {
            transport.wait(slot)?;
            progress(completed);
        }
    }
    Ok(())
}

struct Slot {
    host: *mut c_void,
    begin: CudaEvent,
    end: CudaEvent,
    in_flight: bool,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            host: std::ptr::null_mut(),
            begin: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            in_flight: false,
        }
    }
}

#[derive(Default)]
struct Timing {
    host_alloc_ms: f64,
    cpu_staging_ms: f64,
    ro_cpu_staging_ms: f64,
    dirty_cpu_staging_ms: f64,
    ro_staged_bytes: usize,
    dirty_staged_bytes: usize,
    enqueue_ms: f64,
    completion_wait_ms: f64,
    h2d_event_ms: f64,
    uploaded_bytes: usize,
    clear_enqueue_ms: f64,
    final_sync_ms: f64,
}

#[derive(Clone, Copy)]
enum StagingRegion {
    ReadOnly,
    Dirty,
}

struct PinnedUpload<'a> {
    arena: &'a Arena,
    capacity: usize,
    host_alloc_flags: u32,
    copy_threads: usize,
    region: StagingRegion,
    slots: [Slot; SLOT_COUNT],
    timing: Timing,
}

impl<'a> PinnedUpload<'a> {
    fn new(
        arena: &'a Arena,
        capacity: usize,
        host_alloc_flags: u32,
        copy_threads: usize,
    ) -> Result<Self> {
        let mut owner = Self {
            arena,
            capacity,
            host_alloc_flags,
            copy_threads,
            region: StagingRegion::ReadOnly,
            slots: std::array::from_fn(|_| Slot::default()),
            timing: Timing::default(),
        };
        let started = Instant::now();
        for slot in &mut owner.slots {
            // Store each allocation immediately in the owner so a later error
            // creating events or the second slot cannot leak earlier resources.
            unsafe {
                check(
                    cudaHostAlloc(&mut slot.host, capacity, host_alloc_flags),
                    "initial pinned host alloc",
                )?;
                // cudaHostAlloc leaves bytes uninitialized. Initialize this
                // bounded allocation once before borrowing it as &mut [u8].
                // Staging still writes every copied byte, and real transport
                // never reads the host slot back (including WC allocations).
                std::ptr::write_bytes(slot.host.cast::<u8>(), 0, capacity);
                check(cudaEventCreate(&mut slot.begin), "initial H2D begin event")?;
                check(cudaEventCreate(&mut slot.end), "initial H2D end event")?;
            }
        }
        owner.timing.host_alloc_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(owner)
    }

    fn clear_dirty_region(&mut self) -> Result<()> {
        let started = Instant::now();
        check(
            unsafe {
                cudaMemsetAsync(
                    self.arena.main_dev,
                    0,
                    self.arena.main_size.max(1),
                    self.arena.state.stream,
                )
            },
            "dirty region clear",
        )?;
        self.timing.clear_enqueue_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let started = Instant::now();
        check(
            unsafe { cudaStreamSynchronize(self.arena.state.stream) },
            "initial arena upload sync",
        )?;
        self.timing.final_sync_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}

impl UploadTransport for PinnedUpload<'_> {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn wait(&mut self, index: usize) -> Result<()> {
        let slot = &mut self.slots[index];
        if !slot.in_flight {
            return Err(LlamaError::format("pinned upload waited on an idle slot"));
        }
        let started = Instant::now();
        check(
            unsafe { cudaEventSynchronize(slot.end) },
            "initial H2D completion",
        )?;
        self.timing.completion_wait_ms += started.elapsed().as_secs_f64() * 1000.0;
        // Completion is established before querying timing or reusing memory.
        slot.in_flight = false;
        let mut elapsed = 0.0f32;
        check(
            unsafe { cudaEventElapsedTime(&mut elapsed, slot.begin, slot.end) },
            "initial H2D event timing",
        )?;
        self.timing.h2d_event_ms += f64::from(elapsed);
        Ok(())
    }

    fn stage(&mut self, index: usize, source: &[u8]) -> Result<()> {
        let slot = &mut self.slots[index];
        if slot.in_flight || source.len() > self.capacity {
            return Err(LlamaError::format(
                "pinned upload staging slot is busy or too small",
            ));
        }
        let started = Instant::now();
        // SAFETY: this slot owns capacity bytes of writable CUDA-pinned host
        // memory, initialized once at allocation. Its last DMA read completed
        // before this method; the borrowed
        // Context source remains live and does not overlap the new allocation.
        // The exclusive slice is split into disjoint borrows; no raw pointer is
        // passed to workers, and all workers join before DMA can be enqueued.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(slot.host.cast::<u8>(), source.len()) };
        copy_staging_bytes(source, destination, self.copy_threads)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.timing.cpu_staging_ms += elapsed_ms;
        match self.region {
            StagingRegion::ReadOnly => {
                self.timing.ro_cpu_staging_ms += elapsed_ms;
                self.timing.ro_staged_bytes += source.len();
            }
            StagingRegion::Dirty => {
                self.timing.dirty_cpu_staging_ms += elapsed_ms;
                self.timing.dirty_staged_bytes += source.len();
            }
        }
        Ok(())
    }

    fn enqueue(&mut self, index: usize, destination: usize, len: usize) -> Result<()> {
        let device = self.arena.ptr_at(destination, len)?;
        let stream = self.arena.state.stream;
        let slot = &mut self.slots[index];
        if slot.in_flight || len > self.capacity {
            return Err(LlamaError::format(
                "pinned upload enqueue slot is busy or too small",
            ));
        }
        let started = Instant::now();
        // Mark conservatively before any asynchronous call. In particular a
        // successful copy followed by a failed end-event record must still
        // drain its stream before Drop frees the host source.
        slot.in_flight = true;
        unsafe {
            check(
                cudaEventRecord(slot.begin, stream),
                "initial H2D begin record",
            )?;
            check(
                cudaMemcpyAsync(
                    device,
                    slot.host.cast_const(),
                    len,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream,
                ),
                "initial pinned H2D",
            )?;
            check(cudaEventRecord(slot.end, stream), "initial H2D end record")?;
        }
        self.timing.enqueue_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.timing.uploaded_bytes += len;
        Ok(())
    }
}

impl Drop for PinnedUpload<'_> {
    fn drop(&mut self) {
        if self.slots.iter().any(|slot| slot.in_flight) {
            let stream_error = unsafe { cudaStreamSynchronize(self.arena.state.stream) };
            if stream_error != CUDA_SUCCESS {
                // A failed synchronization does not establish safe reuse/free.
                // Try a device-wide drain. If it also fails, retain the bounded
                // allocations until CUDA context/process teardown rather than
                // risk freeing host memory an unknown DMA operation still reads.
                let device_error = unsafe { cudaDeviceSynchronize() };
                if device_error != CUDA_SUCCESS {
                    tracing::error!(
                        target: "teamy_llm_makepad_gguf::cuda::upload",
                        retained_bytes = self.capacity * SLOT_COUNT,
                        stream_error_code = stream_error,
                        stream_error = %cuda_err_str(stream_error),
                        device_error_code = device_error,
                        device_error = %cuda_err_str(device_error),
                        "Cannot drain pinned staging; retaining allocation until CUDA teardown"
                    );
                    return;
                }
            }
        }
        for slot in &mut self.slots {
            unsafe {
                for event in [slot.begin, slot.end] {
                    if !event.is_null() {
                        let error = cudaEventDestroy(event);
                        if error != CUDA_SUCCESS {
                            tracing::warn!(
                                target: "teamy_llm_makepad_gguf::cuda::upload",
                                error_code = error,
                                error = %cuda_err_str(error),
                                "Initial CUDA upload event cleanup failed"
                            );
                        }
                    }
                }
                if !slot.host.is_null() {
                    let error = cudaFreeHost(slot.host);
                    if error != CUDA_SUCCESS {
                        tracing::warn!(
                            target: "teamy_llm_makepad_gguf::cuda::upload",
                            error_code = error,
                            error = %cuda_err_str(error),
                            "Initial CUDA upload pinned allocation cleanup failed"
                        );
                    }
                }
            }
        }
    }
}

pub(super) fn upload_arena(
    ctx: &Context,
    arena: &Arena,
    progress: &mut dyn FnMut(usize, usize),
    device_alloc_ms: f64,
) -> Result<()> {
    let started = Instant::now();
    let ro_split = ctx.ro_split();
    let dirty_used = ctx.used_mem().saturating_sub(ro_split);
    let total = ro_split.saturating_add(dirty_used);
    let capacity = ro_split.max(dirty_used).clamp(1, MAX_SLOT_BYTES);
    let mut upload = PinnedUpload::new(
        arena,
        capacity,
        process_host_alloc_flags()?,
        process_copy_threads()?,
    )?;
    progress(0, ro_split);
    upload_region(
        &mut upload,
        0,
        ro_split,
        |offset, len| ctx.data_at(offset, len).map_err(LlamaError::format),
        &mut |completed| progress(completed, ro_split),
    )?;
    upload.clear_dirty_region()?;
    upload.region = StagingRegion::Dirty;
    upload_region(
        &mut upload,
        ro_split,
        dirty_used,
        |offset, len| ctx.data_at(offset, len).map_err(LlamaError::format),
        &mut |completed| progress(ro_split + completed, total.max(1)),
    )?;
    upload.finish()?;
    let timing = &upload.timing;
    tracing::debug!(
        target: "teamy_llm_makepad_gguf::cuda::upload",
        mode = "pinned-pipeline",
        bytes = timing.uploaded_bytes,
        slots = SLOT_COUNT,
        slot_bytes = capacity,
        host_alloc_flags = upload.host_alloc_flags,
        copy_threads = upload.copy_threads,
        device_alloc_ms,
        host_alloc_ms = timing.host_alloc_ms,
        cpu_staging_ms = timing.cpu_staging_ms,
        ro_staged_bytes = timing.ro_staged_bytes,
        ro_cpu_staging_ms = timing.ro_cpu_staging_ms,
        dirty_staged_bytes = timing.dirty_staged_bytes,
        dirty_cpu_staging_ms = timing.dirty_cpu_staging_ms,
        enqueue_ms = timing.enqueue_ms,
        completion_wait_ms = timing.completion_wait_ms,
        h2d_completed_event_ms = timing.h2d_event_ms,
        clear_enqueue_ms = timing.clear_enqueue_ms,
        final_sync_ms = timing.final_sync_ms,
        upload_wall_ms = started.elapsed().as_secs_f64() * 1000.0,
        "Initial CUDA arena upload completed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn write_combined_is_an_explicit_experiment_without_changing_the_default() {
        assert_eq!(experimental_host_alloc_flags(None).unwrap(), 0);
        assert_eq!(experimental_host_alloc_flags(Some("0")).unwrap(), 0);
        assert_eq!(experimental_host_alloc_flags(Some("1")).unwrap(), 4);
        for value in ["", "true", "false", "4", " 1", "1 ", "-1"] {
            assert!(experimental_host_alloc_flags(Some(value)).is_err());
        }
    }

    #[test]
    fn copy_threads_are_an_explicit_bounded_experiment() {
        assert_eq!(experimental_copy_threads(None).unwrap(), 1);
        for (value, count) in [("1", 1), ("2", 2), ("4", 4)] {
            assert_eq!(experimental_copy_threads(Some(value)).unwrap(), count);
        }
        for value in ["", "0", "3", "8", "true", " 2", "2 ", "-1"] {
            assert!(experimental_copy_threads(Some(value)).is_err());
        }
    }

    #[test]
    fn parallel_staging_preserves_exact_bytes_odd_partitions_and_tail_guards() {
        for len in [
            0,
            1,
            PARALLEL_COPY_MIN_BYTES - 1,
            PARALLEL_COPY_MIN_BYTES,
            PARALLEL_COPY_MIN_BYTES + 1,
            PARALLEL_COPY_MIN_BYTES + 7,
            3 * PARALLEL_COPY_MIN_BYTES + 3,
        ] {
            let source: Vec<u8> = (0..len).map(|index| (index * 31) as u8).collect();
            for threads in [1, 2, 4] {
                let mut output = vec![0xcd; len + 6];
                copy_staging_bytes(&source, &mut output[3..3 + len], threads).unwrap();
                assert_eq!(&output[..3], &[0xcd; 3]);
                assert_eq!(&output[3..3 + len], source);
                assert_eq!(&output[3 + len..], &[0xcd; 3]);
            }
        }
    }

    #[test]
    fn small_staging_is_sequential_and_invalid_shapes_do_not_copy() {
        let source = [1, 2, 3, 4, 5];
        let mut destination = [0; 5];
        let current_thread = std::thread::current().id();
        copy_staging_bytes_with(
            &source,
            &mut destination,
            4,
            &|_| panic!("small copy must not attempt worker creation"),
            &|index, source, destination| {
                assert_eq!(index, 0);
                assert_eq!(std::thread::current().id(), current_thread);
                destination.copy_from_slice(source);
            },
        )
        .unwrap();
        assert_eq!(destination, source);
        for threads in [0, 3, 5, 8] {
            assert!(copy_staging_bytes(&source, &mut destination, threads).is_err());
        }
        let mut too_short = [0xcd; 4];
        assert!(copy_staging_bytes(&source, &mut too_short, 1).is_err());
        assert_eq!(too_short, [0xcd; 4]);
    }

    #[test]
    fn spawn_failure_joins_every_started_worker_before_returning() {
        let source = vec![0x37; PARALLEL_COPY_MIN_BYTES + 7];
        let mut destination = vec![0; source.len()];
        let released = AtomicBool::new(false);
        let completed = AtomicUsize::new(0);
        let result = copy_staging_bytes_with(
            &source,
            &mut destination,
            4,
            &|index| {
                if index == 2 {
                    released.store(true, Ordering::Release);
                    Err(std::io::Error::other("injected spawn failure"))
                } else {
                    Ok(())
                }
            },
            &|_, source, destination| {
                // An unexpected OS spawn failure before the injected failure
                // must fail this fixture rather than strand a borrowed worker.
                let deadline = Instant::now() + std::time::Duration::from_secs(5);
                while !released.load(Ordering::Acquire) {
                    assert!(
                        Instant::now() < deadline,
                        "fixture copy worker was not released before its deadline"
                    );
                    std::thread::yield_now();
                }
                destination.copy_from_slice(source);
                completed.fetch_add(1, Ordering::Release);
            },
        );
        assert!(result.unwrap_err().to_string().contains("worker 2 failed"));
        assert_eq!(completed.load(Ordering::Acquire), 2);
        let copied = 2 * source.len().div_ceil(4);
        assert_eq!(&destination[..copied], &source[..copied]);
        assert!(destination[copied..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn worker_panic_still_joins_all_other_borrowed_destinations() {
        let source = vec![0x68; PARALLEL_COPY_MIN_BYTES + 3];
        let mut destination = vec![0; source.len()];
        let released = AtomicBool::new(false);
        let completed = AtomicUsize::new(0);
        let result = copy_staging_bytes_with(
            &source,
            &mut destination,
            4,
            &|_| Ok(()),
            &|index, source, destination| {
                if index == 0 {
                    released.store(true, Ordering::Release);
                    panic!("injected staging worker panic");
                }
                while !released.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                destination.copy_from_slice(source);
                completed.fetch_add(1, Ordering::Release);
            },
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("worker 0 panicked"));
        assert_eq!(completed.load(Ordering::Acquire), 3);
        let uncopied = source.len().div_ceil(4);
        assert!(destination[..uncopied].iter().all(|byte| *byte == 0));
        assert_eq!(&destination[uncopied..], &source[uncopied..]);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Action {
        Stage(usize),
        Enqueue(usize, usize, usize),
        Wait(usize),
        DrainOnDrop,
    }

    struct FakeTransport {
        capacity: usize,
        host: [Vec<u8>; SLOT_COUNT],
        pending: [Option<(usize, usize)>; SLOT_COUNT],
        output: Rc<RefCell<Vec<u8>>>,
        actions: Rc<RefCell<Vec<Action>>>,
        fail_after_enqueue: Option<usize>,
        enqueue_count: usize,
    }

    impl FakeTransport {
        fn new(capacity: usize, output: Rc<RefCell<Vec<u8>>>) -> Self {
            Self {
                capacity,
                host: std::array::from_fn(|_| Vec::new()),
                pending: [None; SLOT_COUNT],
                output,
                actions: Rc::new(RefCell::new(Vec::new())),
                fail_after_enqueue: None,
                enqueue_count: 0,
            }
        }

        fn complete(&mut self, slot: usize) {
            if let Some((destination, len)) = self.pending[slot].take() {
                // Read the CURRENT staging bytes only at DMA completion. A
                // scheduler that reused a slot early would corrupt the result.
                self.output.borrow_mut()[destination..destination + len]
                    .copy_from_slice(&self.host[slot]);
            }
        }
    }

    impl UploadTransport for FakeTransport {
        fn capacity(&self) -> usize {
            self.capacity
        }

        fn wait(&mut self, slot: usize) -> Result<()> {
            assert!(self.pending[slot].is_some());
            self.actions.borrow_mut().push(Action::Wait(slot));
            self.complete(slot);
            Ok(())
        }

        fn stage(&mut self, slot: usize, source: &[u8]) -> Result<()> {
            assert!(self.pending[slot].is_none(), "staged over in-flight DMA");
            assert!(source.len() <= self.capacity);
            self.actions.borrow_mut().push(Action::Stage(slot));
            self.host[slot] = source.to_vec();
            Ok(())
        }

        fn enqueue(&mut self, slot: usize, destination: usize, len: usize) -> Result<()> {
            assert!(self.pending[slot].is_none());
            self.actions
                .borrow_mut()
                .push(Action::Enqueue(slot, destination, len));
            self.pending[slot] = Some((destination, len));
            self.enqueue_count += 1;
            if self.fail_after_enqueue == Some(self.enqueue_count) {
                return Err(LlamaError::format(
                    "injected completion-event failure after DMA enqueue",
                ));
            }
            Ok(())
        }
    }

    impl Drop for FakeTransport {
        fn drop(&mut self) {
            if self.pending.iter().any(Option::is_some) {
                self.actions.borrow_mut().push(Action::DrainOnDrop);
                for slot in 0..SLOT_COUNT {
                    self.complete(slot);
                }
            }
        }
    }

    #[test]
    fn upload_preserves_every_byte_partial_chunks_offsets_and_completed_progress() {
        let capacity = 7;
        for len in [0, 1, 6, 7, 8, 14, 15, 21, 22, 35] {
            let start = 3;
            let source: Vec<u8> = (0..start + len).map(|i| (i * 31) as u8).collect();
            let output = Rc::new(RefCell::new(vec![0xcc; start + len + 2]));
            let mut fake = FakeTransport::new(capacity, output.clone());
            let mut completed = Vec::new();
            upload_region(
                &mut fake,
                start,
                len,
                |offset, count| Ok(&source[offset..offset + count]),
                &mut |count| completed.push(count),
            )
            .unwrap();
            assert_eq!(&output.borrow()[start..start + len], &source[start..]);
            assert_eq!(&output.borrow()[..start], &[0xcc; 3]);
            assert_eq!(&output.borrow()[start + len..], &[0xcc; 2]);
            assert!(completed.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(completed.last().copied(), (len != 0).then_some(len));
            assert!(fake.pending.iter().all(Option::is_none));
        }
    }

    #[test]
    fn stages_next_slot_before_waiting_and_waits_before_reusing_either_slot() {
        let source = [1, 2, 3, 4, 5, 6, 7];
        let output = Rc::new(RefCell::new(vec![0; source.len()]));
        let mut fake = FakeTransport::new(3, output);
        upload_region(
            &mut fake,
            0,
            source.len(),
            |offset, count| Ok(&source[offset..offset + count]),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            *fake.actions.borrow(),
            [
                Action::Stage(0),
                Action::Enqueue(0, 0, 3),
                Action::Stage(1),
                Action::Enqueue(1, 3, 3),
                Action::Wait(0),
                Action::Stage(0),
                Action::Enqueue(0, 6, 1),
                Action::Wait(1),
                Action::Wait(0),
            ],
        );
    }

    #[test]
    fn failed_completion_event_after_enqueue_keeps_source_alive_for_drop_drain() {
        let source = [1, 2, 3, 4, 5, 6];
        let output = Rc::new(RefCell::new(vec![0; source.len()]));
        let actions;
        {
            let mut fake = FakeTransport::new(3, output.clone());
            fake.fail_after_enqueue = Some(2);
            actions = fake.actions.clone();
            assert!(upload_region(
                &mut fake,
                0,
                source.len(),
                |offset, count| Ok(&source[offset..offset + count]),
                &mut |_| {},
            )
            .is_err());
            assert!(fake.pending.iter().all(Option::is_some));
            assert_eq!(*output.borrow(), [0; 6]);
        }
        assert_eq!(*output.borrow(), source);
        assert_eq!(actions.borrow().last(), Some(&Action::DrainOnDrop));
    }

    #[test]
    fn source_failure_or_cancellation_does_not_discard_pending_staging() {
        let source = [1, 2, 3, 4, 5, 6];
        let output = Rc::new(RefCell::new(vec![0; source.len()]));
        {
            let mut fake = FakeTransport::new(3, output.clone());
            assert!(upload_region(
                &mut fake,
                0,
                source.len(),
                |offset, count| {
                    if offset == 3 {
                        Err(LlamaError::format(
                            "injected cancellation before source acquisition",
                        ))
                    } else {
                        Ok(&source[offset..offset + count])
                    }
                },
                &mut |_| {},
            )
            .is_err());
            assert!(fake.pending[0].is_some());
        }
        assert_eq!(&output.borrow()[..3], &source[..3]);
        assert_eq!(&output.borrow()[3..], &[0; 3]);
    }

    #[test]
    fn callback_unwind_keeps_other_pending_slot_alive_until_drop() {
        let source = [1, 2, 3, 4, 5, 6, 7];
        let output = Rc::new(RefCell::new(vec![0; source.len()]));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut fake = FakeTransport::new(3, output.clone());
            let _ = upload_region(
                &mut fake,
                0,
                source.len(),
                |offset, count| Ok(&source[offset..offset + count]),
                &mut |_| panic!("injected progress callback failure"),
            );
        }));
        assert!(result.is_err());
        assert_eq!(&output.borrow()[..6], &source[..6]);
        assert_eq!(output.borrow()[6], 0);
    }
}
