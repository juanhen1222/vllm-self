# HF3FS KV Connector 系统设计

## 概述

HF3FS KV Connector 是 vLLM 中基于 3FS 的 KV 缓存分布式存储解决方案，通过前缀哈希匹配和异步 I/O 实现跨请求缓存复用。

## 系统架构

```
Scheduler                Worker                 Metadata Server
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│Token Prefix│◄────────┤AsyncOperation│◄────────┤ Global Key  │
│  Matching   │         │   Manager   │         │  Metadata   │
└─────────────┘         └─────────────┘         └─────────────┘
      │                        │                        │
      └────────────────────────┼────────────────────────┘
                                │
                        ┌───────▼───────┐
                        │  3FS Storage  │
                        └───────────────┘
```

## 核心流程

### Scheduler 侧

#### 1. get_num_new_matched_tokens - 缓存匹配
```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # 生成前缀哈希序列
    block_hashes = self._generate_block_hashes(request.prompt_token_ids)
    
    # 批量检查块存在性
    exists_results = self._metadata_client.batch_key_exists(block_hashes)
    
    # 计算连续前缀块的匹配数
    matched_blocks = count_consecutive_matches(exists_results)
    return max(0, matched_blocks * block_size - num_computed_tokens)
```

#### 2. update_state_after_alloc - 状态更新
```python
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    state = self._get_scheduling_state(request.request_id)
    # 保存需要 Fetch 的 Block 的 BlockIndex
    state.load_op.need_fetch_block_ids = blocks.get_unhashed_block_ids()
    state.phase = "WAITING_TO_LOAD"
```

#### 3. build_connector_meta - 元数据构建
```python
def build_connector_meta(self, scheduler_output):
    # 分阶段处理：Waiting For Load -> New Request -> Cached Request
    for each_phase in [waiting_to_load, new_requests, cached_requests]:
        process_requests(phase, metadata)
    return metadata
```

### Worker 侧

#### 1. start_load_kv - 异步加载
```python
def start_load_kv(self, forward_context):
    for request in metadata.requests:
        if request.load_block_op:
            # 批量提交加载任务到后台线程
            self._async_manager.submit_load_operation(
                request.request_id, block_ids, block_hashes
            )
```

#### 2. wait_for_save - 异步保存
```python
def wait_for_save(self):
    for request in metadata.requests:
        if request.save_block_op:
            # 跳过已缓存的块，只保存新生成的
            skip_blocks = request.save_block_op.skip_leading_blocks
            new_block_hashes = generate_hashes(request.tokens[skip_blocks:])
            self._async_manager.submit_save_operation(request.request_id, block_ids, new_block_hashes)
```

#### 3. get_finished - 状态查询
```python
def get_finished(self, finished_req_ids):
    # 检查异步操作完成状态
    completed_saves, completed_loads = self._async_manager.get_finished_operations()
    return completed_saves, completed_loads
```

## 异步操作管理

### AsyncOperationManager 核心机制

```python
class AsyncOperationManager:
    # 后台工作线程处理 I/O
    def _save_worker(self):
        while task := self._save_queue.get():
            # 1. 分配存储页面
            pages = metadata_client.allocate_pages_for_keys(block_hashes)
            # 2. 通过 triton gather all layer's kvcache into buffers
            with torch.cuda.stream(save_stream):
                gather_kv_caches(block_ids, buffers)
            # 3. 并发写入 3FS
            clients.batch_write(offsets, buffers)
            # 4. 确认写入成功
            metadata_client.confirm_write_for_keys(written_keys)
    
    def _load_worker(self):
        while task := self._load_queue.get():
            # 1. 查询块位置
            page_indices = metadata_client.get_key_locations(block_hashes)
            # 2. 异步读取数据
            clients.batch_read(offsets, buffers)
            # 3. 通过 triton scatter buffers into layer's kvcache
            with torch.cuda.stream(load_stream):
                scatter_kv_caches(block_ids, buffers)
```