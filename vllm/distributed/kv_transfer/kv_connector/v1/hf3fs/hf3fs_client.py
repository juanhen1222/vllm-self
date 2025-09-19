import logging
import multiprocessing
import os
import threading
from functools import wraps
from pathlib import Path
from typing import List

import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import load

root = Path(__file__).parent.resolve()
cuda_include_path = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
hf3fs_utils = load(
    name="hf3fs_utils",
    sources=[f"{root}/utils/hf3fs_utils.cpp"],
    extra_include_paths=[cuda_include_path],
)

logger = logging.getLogger(__name__)

HF3FS_AVAILABLE = True
try:
    from hf3fs_fuse.io import (
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except ImportError:
    HF3FS_AVAILABLE = False


def rsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.rlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


def wsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.wlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


class Hf3fsClient:
    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):
        """Initialize the HF3FS client with hf3fs_fuse.

        Args:
            path: Path to the file used for storage
            size: Total size of the storage file in bytes
            bytes_per_page: Size of each page in bytes
            entries: Maximum number of concurrent operations
        """
        if not HF3FS_AVAILABLE:
            raise ImportError(
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package."
            )

        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries

        # Create the file if it doesn't exist and set its size
        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.file, size)
        register_fd(self.file)

        self.hf3fs_mount_point = extract_mount_point(path)
        self.bs = self.bytes_per_page
        self.shm_r = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )
        self.shm_w = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )

        self.shm_r_tensor = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
        self.shm_w_tensor = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)

        numel = self.bs * self.entries
        self.r_pinned = torch.empty(
            numel,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self.w_pinned = torch.empty(
            numel,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )

        self.numa = -1
        self.ior_r = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=True,
            timeout=1,
            numa=self.numa,
        )
        self.ior_w = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=False,
            timeout=1,
            numa=self.numa,
        )
        self.iov_r = make_iovec(self.shm_r, self.hf3fs_mount_point)
        self.iov_w = make_iovec(self.shm_w, self.hf3fs_mount_point)
        self.shm_r.unlink()
        self.shm_w.unlink()

        self.rlock = threading.RLock()
        self.wlock = threading.RLock()

        self.stream = torch.cuda.Stream()
        self.stream_ptr_int = self.stream.cuda_stream

        logger.debug(f"Initialized HF3FS client with file: {path}, size: {size} bytes")

    @rsynchronized()
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        """Read data from the file at specified offsets into tensors.

        Args:
            offsets: List of byte offsets to read from
            tensors: List of tensors to read data into

        Returns:
            List of operation results (0 for success, non-zero for error)
        """
        self.check(offsets, tensors)

        # prepare
        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            self.ior_r.prepare(
                self.iov_r[current : current + size], True, self.file, offset
            )
            current += size

        # submit
        ionum = len(offsets)
        resv = self.ior_r.submit().wait(min_results=ionum)

        # results
        with torch.cuda.stream(self.stream):
            hf3fs_utils.read_shm(
                self.shm_r_tensor, self.r_pinned, tensors, self.stream_ptr_int
            )
        results = [res.result for res in resv]

        return results

    @wsynchronized()
    def batch_write(
        self, offsets: List[int], tensors: List[torch.Tensor], event: torch.cuda.Event
    ) -> List[int]:
        """Write data from tensors to the file at specified offsets.

        Args:
            offsets: List of byte offsets to write to
            tensors: List of tensors containing data to write

        Returns:
            List of operation results (0 for success, non-zero for error)
        """

        self.check(offsets, tensors)
        # prepare
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(event)
            hf3fs_utils.write_shm(
                tensors, self.shm_w_tensor, self.w_pinned, self.stream_ptr_int
            )

        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            self.ior_w.prepare(
                self.iov_w[current : current + size], False, self.file, offset
            )
            current += size

        # submit
        ionum = len(offsets)
        resv = self.ior_w.submit().wait(min_results=ionum)

        # results
        results = [res.result for res in resv]

        return results

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:
        sizes = [t.numel() * t.itemsize for t in tensors]
        if any(
            [
                len(offsets) > self.entries,
                len(offsets) != len(sizes),
                all(
                    [
                        offset < 0 or offset + size > self.size
                        for offset, size in zip(offsets, sizes)
                    ]
                ),
                all([size > self.bytes_per_page for size in sizes]),
            ]
        ):
            self.close()
            raise ValueError(f"Hf3fsClient.check Failed")

    def get_size(self) -> int:
        """Get the total size of the storage file.

        Returns:
            Size of the file in bytes
        """
        return self.size

    def close(self) -> None:
        """Close the client and clean up resources."""
        deregister_fd(self.file)
        os.close(self.file)
        del self.ior_r
        del self.ior_w
        del self.iov_r
        del self.iov_w
        self.shm_r.close()
        self.shm_w.close()

    def flush(self) -> None:
        """Flush any pending writes to disk."""
        os.fsync(self.file)
