# Nemotron Nano 12B v2 VL — SFT on LLaVA-Instruct-150K (v0.5.0 clean clone)

이 문서는 **clean한 NeMo-RL v0.5.0 위에 LLaVA-Instruct-150K로 Nemotron Nano 12B v2 VL SFT**를 돌리기 위한 완전 핸드오프입니다. fork (`/ephemeral/vuvlm/nemo-RL`)와 완전 분리되어 동작합니다.

검증: 2026-04-21, 1×H100 PCIe, LoRA(dim=8) SFT, 200 샘플 / 20 step 완주, val_loss 11.7913 → 11.7884.

---

## 1. 사전 조건

호스트 / 마운트는 fork의 `/ephemeral/vuvlm/nemo-RL/DOCKER_VUVLM.md` 와 같음 (8×H100 PCIe brev pod, `/ephemeral` bind, `vuvlm:nemo-rl-0.5.0` 이미지 이미 빌드됨).

확인:
```bash
docker images | grep vuvlm:nemo-rl-0.5.0
ls /ephemeral/vuvlm/nemo-RL-v0.5.0          # 클론 존재 여부
ls /ephemeral/vuvlm/sft-data/llava_smoke    # 스모크 데이터 존재 여부
```

없으면 §7 부록의 "최초 설치" 절차로.

---

## 2. 컨테이너 기동

이 환경은 **fork의 `vuvlm` 컨테이너와 완전 분리된 별도 컨테이너 `vuvlm-v050`** 를 씁니다 (compose 파일도 다름).

```bash
cd /ephemeral/vuvlm
docker compose -f docker-compose.v050.yml up -d
docker ps                       # vuvlm-v050 떠 있는지 확인
```

들어가서 작업할 때:
```bash
docker exec -it vuvlm-v050 bash
# 컨테이너 내부 cwd = /ephemeral/vuvlm/nemo-RL-v0.5.0
```

> **주의**: `vuvlm`과 `vuvlm-v050` 둘 다 동시 기동은 가능하지만, **학습은 한 번에 하나만** (둘 다 GPU 8장 잡으려고 함).

---

## 3. 스모크 학습 실행

검증된 한 줄:
```bash
docker exec -d vuvlm-v050 bash -lc \
  "cd /ephemeral/vuvlm/nemo-RL-v0.5.0 && \
   uv run python examples/run_vlm_sft.py \
     --config examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml \
     > /ephemeral/vuvlm/nemo-RL-v0.5.0/logs/smoke_$(date +%Y%m%d_%H%M%S).log 2>&1"
```

- `-d` 로 백그라운드 실행 → 컨테이너에서 빠져나와도 계속 돌아감.
- `uv run` 필수 (직접 `/opt/nemo_rl_venv/bin/python` 쓰면 NeMo-RL 내부 venv 부트스트랩이 깨짐).
- 단일 GPU LoRA 기준 ~5분 안에 20 step 완주.

진행 확인:
```bash
LOG=$(ls -t /ephemeral/vuvlm/nemo-RL-v0.5.0/logs/smoke_*.log | head -1)
tail -F "$LOG" | grep -E "Step |Validation|Error|Killed"
```

---

## 4. Loss 그래프 모니터링 (TensorBoard)

학습 시작하면 자동으로 `nemo-RL-v0.5.0/logs/exp_NNN/tensorboard/` 에 TFEvents가 쌓입니다 (config의 `logger.tensorboard_enabled: true`).

### 4.1 TensorBoard 서버 띄우기

이미지 안에 TensorBoard 깔려 있으니 컨테이너에서 바로 실행:

```bash
docker exec -d vuvlm-v050 bash -lc \
  "uv run tensorboard \
     --logdir /ephemeral/vuvlm/nemo-RL-v0.5.0/logs \
     --bind_all --port 6006 \
     > /tmp/tb.log 2>&1"
```

서버 떴는지 확인:
```bash
docker exec vuvlm-v050 cat /tmp/tb.log    # "TensorBoard 2.x.x at http://..." 한 줄 보여야 함
```

### 4.2 브라우저로 접속

**옵션 A — Brev 포트 포워딩 (권장):**
```bash
# 호스트(brev pod 셸)에서:
brev port-forward 6006:6006        # brev CLI 인스톨돼 있을 때
# 또는 SSH 터널:
ssh -L 6006:localhost:6006 <brev-pod>
```
브라우저에서 `http://localhost:6006`.

**옵션 B — 컨테이너 포트 매핑 (호스트에서 바로 보고 싶을 때):**
`docker-compose.v050.yml` 에 `ports: ["6006:6006"]` 추가하고 `docker compose -f docker-compose.v050.yml up -d` 다시.

### 4.3 보면 좋은 메트릭

TensorBoard 좌측 트리에서 **`train/`** 와 **`validation/`** 그룹 둘 다 펼쳐서:
- `train/loss` — step당 로스
- `train/grad_norm` — 발산 모니터링
- `validation/val_loss` — 매 `val_period` step마다
- `timing/train/*` — 처리량 (특히 `valid_tokens_per_sec_per_gpu`)

### 4.4 TensorBoard 쓰기 싫으면 — 텍스트 파싱

```bash
LOG=/ephemeral/vuvlm/nemo-RL-v0.5.0/logs/smoke_<ts>.log
grep -E "Validation loss|Loss:" $LOG
```

---

## 4.5 Checkpoint 저장 (위치 / 구조 / 재개)

### 4.5.1 Config

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: "/ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora"
  metric_name: "train:loss"     # val 끄면 train:loss, val 살리면 val:val_loss
  higher_is_better: false
  keep_top_k: 3                 # best 3개 step만 유지, 나머지 자동 삭제
  save_period: 10               # N step마다 저장 시도
  checkpoint_must_save_by: null # 절대 시간 마감 강제 (예: 학습 잡 종료 5분전 강제 save)
```

> `checkpoint_dir`은 **절대 경로 권장**. default `"results/sft_${policy.model_name}"` 는 model_name에 슬래시(`nvidia/...`) 들어가서 디렉토리 두 단계로 갈라짐.

### 4.5.2 디스크 레이아웃

저장 step마다 다음 구조 생성:

```
results/sft_nemotron_vl_lora/
├── step_10/                          # save_period 마다 한 디렉토리
│   ├── config.yaml                   # 그 시점 final config 스냅샷 (재개·재현용)
│   ├── training_info.json            # step / consumed_samples / metric 등 상태
│   ├── train_dataloader.pt           # 데이터로더 RNG / 위치 (재개 시 이어 읽기)
│   └── policy/
│       ├── weights/
│       │   └── model/
│       │       ├── .metadata         # DCP (PyTorch Distributed Checkpoint) 메타
│       │       └── __<rank>_0.distcp # rank 별 shard (8 rank → 8 파일)
│       └── optimizer/
│           └── optim/
│               ├── .metadata
│               └── __<rank>_0.distcp # optimizer state shards
├── step_20/...
└── step_30/...
```

**Tmp 디렉토리**: 저장 중에는 `tmp_step_N/` 으로 쓰고, 다 쓰면 atomically `step_N/` 으로 rename. 학습 중 `tmp_step_N/` 보이면 **저장 진행 중** 의미. 끝까지 안 마무리되면 (예: 도중에 죽음) `tmp_step_N/` 로 남고, 재개 시 무시됨 — 안전.

**용량 추정 (12B + LoRA dim=8, 8 rank)**: step당 약 **350 MB** 측정. `keep_top_k=3` 이면 최대 ~1 GB 점유. 한 step에서 다음으로 넘어가는 동안 best 아닌 옛 step은 자동 삭제됨.

> base 모델은 frozen (LoRA) 이므로 weights 안에 base까지 통째로 들어가지 않음. 만약 full FT 라면 12B × 2 bytes × 1.5 (옵티마이저까지) = 약 **40 GB / step** 됨 — `keep_top_k`와 디스크 잘 맞춰야.

### 4.5.3 학습 재개 (resume)

```bash
docker exec -d vuvlm-v050 bash -lc \
  "cd /ephemeral/vuvlm/nemo-RL-v0.5.0 && \
   uv run python examples/run_vlm_sft.py \
     --config examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml \
     checkpointing.checkpoint_dir=/ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora \
     > logs/resume_$(date +%H%M%S).log 2>&1"
```

NeMo-RL이 `checkpoint_dir` 안의 가장 최근 `step_N/` 자동 인식 → 그 지점부터 이어서 학습. config 안 바꾸고 같은 디렉토리로 재실행하면 재개됨.

처음부터 다시 돌리고 싶으면:
- `checkpoint_dir` 다른 경로로 바꾸거나
- 기존 디렉토리 비우기 (`rm -rf results/sft_nemotron_vl_lora/step_*`)

### 4.5.4 추론에 쓰기 — HuggingFace 형식 변환

DCP `__<rank>_0.distcp` shard는 **HuggingFace `from_pretrained()` 로 직접 못 읽음**. Consolidate 필요. fork에서 검증된 방법:

config에 `policy.dtensor_cfg.single_rank_consolidation: true` 추가하면 저장 시 자동으로 `step_N/policy/weights/model/consolidated/` 안에 HF-호환 7-shard safetensors 생성됨 (fork의 `consolidation_fix` 메모리 참조; ~25분 추가 시간).

또는 사후 변환: NeMo-RL `tools/convert_dcp_to_hf.py` 비슷한 스크립트가 있을 가능성 (확인 필요). 없으면 fork 트리에 있을 수도 있음.

LoRA만 쓸 거면 더 간단 — NeMo-RL의 LoRA 저장 경로(`step_N/policy/weights/model/`)는 이미 PEFT-호환 포맷(`adapter_config.json` + `adapter_model.safetensors`)이라 바로 `PeftModel.from_pretrained()` 로 얹어서 inference 가능. 자세한 pipeline (merge → vLLM multi-GPU) 은 **§6** 참조.

---

## 5. 자기 데이터 / 본 학습으로 확장

### 5.1 데이터 늘리기

스모크 데이터(200개)는 `make_llava_smoke.py` 가 `liuhaotian/LLaVA-Instruct-150K` 에서 추출한 것:
```bash
docker exec -it vuvlm-v050 bash
cd /ephemeral/vuvlm/sft-data
# 예: 50K train + 5K val 로 확장
python make_llava_smoke.py \
   --out /ephemeral/vuvlm/sft-data/llava_full \
   --n-train 50000 --n-val 5000 --workers 32
```

이미지 다운로드 (COCO train2017) 가 IO bound — 50K면 평균 20-40분 정도.

### 5.2 config 새로 만들기

`examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml` 을 복사해서:
- `data.train_data_path` / `data.val_data_path` → 새 경로
- `sft.max_num_steps` 늘리기 (예: 1000)
- `sft.val_period`, `checkpointing.enabled: true`, `keep_top_k`, `save_period` 도 조정
- `policy.train_global_batch_size` — multi-GPU 활성화 후에만 의미 있음 (지금 1 GPU 기준 1)

### 5.3 Multi-GPU 활성화 (TODO)

이번 검증은 1 GPU 였습니다. 8 GPU FSDP 는 아래 이슈 미해결:
- 호스트 GPU 토폴로지: pair (0-1, 2-3, 4-5, 6-7) 안에서만 NVLink, pair 간은 PCIe → cross-pair P2P/SHM 모두 NCCL CUDA error 217 (`peer access not supported`).
- 권장 다음 시도: `policy.dtensor_cfg.tensor_parallel_size: 2` 로 NVLink-pair 안에서 TP, pair 간은 DP. 그래도 cross-pair NCCL collective 필요는 남음 (FSDP all-gather/reduce-scatter). 별 워크스트림 필요.

---

## 6. 추론 (vLLM, merged LoRA)

LoRA 학습 후 체크포인트를 **base 에 merge → vLLM multi-GPU 서빙** 하는 pipeline. 2026-04-21 검증: `step_10` adapter + LLaVA val 5 샘플, TP=2 (GPU 0-1 NVLink pair), greedy, max_new=8192, 전부 EOS 정상 종료.

두 스크립트:
- `tools/inference/merge_lora_nemotron_vl.py` — PEFT adapter + base → merged safetensors 체크포인트
- `tools/inference/vllm_infer_val.py` — merged 체크포인트를 vLLM 로 TP=2 서빙, validation jsonl 샘플 추론 → jsonl 저장

### 6.1 전제

- 학습이 `adapter_config.json` / `adapter_model.safetensors` 를 `results/.../step_N/policy/weights/model/` 아래에 저장해 둔 상태 (LoRA SFT 정상 종료 시 자동).
- `dtensor_v2` worker venv (vLLM·transformers·timm·open_clip 구비) 에 **peft** 추가 설치:
  ```bash
  docker exec vuvlm-v050 bash -lc '
  DT=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
  # --no-deps --target 중요 (--target 안 쓰면 peft가 새 torch/transformers를 끌어와 기존 환경을 부수고,
  #                        --no-deps 안 쓰면 --target 위에 torch/cuda 라이브러리가 통째로 재설치됨)
  $DT/bin/pip install --no-deps --target $DT/lib/python3.12/site-packages peft
  '
  ```
  설치 확인: `import peft; print(peft.__version__)` 이 **0.19.1** 나오면 OK.

### 6.2 LoRA → base 에 merge

```bash
docker exec -d vuvlm-v050 bash -lc '
export CUDA_VISIBLE_DEVICES=0
DT=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
$DT/bin/python tools/inference/merge_lora_nemotron_vl.py \
  --adapter-dir /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/step_10/policy/weights/model \
  --out-dir     /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/step_10_merged \
  --device cuda:0 \
  > /ephemeral/vuvlm/nemo-RL-v0.5.0/logs/merge_step10_$(date +%Y%m%d_%H%M%S).log 2>&1
'
```

- 1 GPU 로 충분 (12B bf16 로드 ~24 GB). 전체 merge+save ~1 분.
- 결과: `step_10_merged/` 안에 `model-0000{1..6}-of-00006.safetensors` 등 **25 GB** 규모 HF 호환 체크포인트 + `modeling.py` 등 custom code 복사.

**주의 — torchao 호환 패치**: peft 0.19 의 `dispatch_torchao` 가 torchao<0.16 에서 `ImportError` 를 던집니다. 우리 LoRA 는 일반 `Linear` 이라 torchao 자체가 필요 없지만, dispatcher 는 모든 모듈에서 한 번씩 호출됩니다. 그래서 스크립트 상단에서 두 군데 monkey-patch:
```python
import peft.import_utils as _peft_imp
_peft_imp.is_torchao_available = lambda: False
from peft.tuners.lora import torchao as _lora_torchao
_lora_torchao.is_torchao_available = lambda: False
```
peft 의 minor bump / torchao >= 0.16 이 생기면 지워도 됨.

### 6.3 vLLM multi-GPU inference

```bash
docker exec -d vuvlm-v050 bash -lc '
export CUDA_VISIBLE_DEVICES=0,1   # NVLink pair 1개 선택. 2,3 / 4,5 / 6,7 도 OK. 0,2 등 pair-간은 NCCL 217
DT=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
$DT/bin/python tools/inference/vllm_infer_val.py \
  --model-dir  /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/step_10_merged \
  --val-jsonl  /ephemeral/vuvlm/sft-data/llava_smoke/llava_smoke_val.jsonl \
  --n 5 --tp 2 --max-new-tokens 8192 \
  --out-jsonl  /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/inference/val_step10.jsonl \
  > /ephemeral/vuvlm/nemo-RL-v0.5.0/logs/vllm_infer_$(date +%Y%m%d_%H%M%S).log 2>&1
'
```

스크립트 동작:
- `val_jsonl` 에서 첫 N 개 샘플을 읽고, 각 샘플의 **첫 user turn만** (image + text) 프롬프트로 사용.
- `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 로 generation 프롬프트 구성. 시스템 메시지 없음 (학습 데이터가 시스템 없이 user 로 시작했으므로 매칭).
- `llm.generate([{"prompt": ..., "multi_modal_data": {"image": PIL}}, ...], SamplingParams(temperature=0))` 배치.
- 출력 jsonl 한 행당: `image_path`, `user_text`, `gt`, `prediction`, `prompt_text`, `finish_reason`, `prompt_token_ids_len`, `output_token_ids_len`.

**vLLM 튜닝 포인트**:
- `tensor_parallel_size=2`: NVLink pair 안에서만. pair-간 TP 는 NCCL 217 로 실패 (학습과 동일 제약).
- `dtype=bfloat16`, `gpu_memory_utilization=0.85`.
- `enforce_eager=True`: Mamba hybrid + CUDA graph 미검증이라 eager 우선. 안정되면 꺼서 속도 이득 보면 됨.
- `limit_mm_per_prompt={"image": 1}`: 현재 val 은 이미지 1 장/턴. 멀티-이미지 확장 시 올려야 함.
- `max_model_len=16384`: 8K 이미지 토큰 + 8K 텍스트 충분. 필요시 조정.

### 6.4 검증 스냅샷 (2026-04-21, step_10)

| img | in tok | out tok | finish | 요약 |
|---|---|---|---|---|
| 000000361332.jpg | 3354 | 128 | stop | GT "tree branch" vs PRED 장황하지만 일치 |
| 000000330295.jpg | 1819 | 107 | stop | GT "전화+담배" vs PRED 일치 |
| 000000571245.jpg | 1819 | 110 | stop | GT "ski jump" vs PRED 일치 |
| 000000433480.jpg | 3356 | 76  | stop | GT "walking" vs PRED "standing still" 차이 |
| 000000431847.jpg | 3356 | 86  | stop | GT "두 마리" vs PRED "세 마리" 불일치 |

- weights 로드 5.24 s, 5-prompt generate ~수 초.
- step_10 LoRA 는 사실상 base 모델 그대로 (val_loss 11.79 → 11.79, 학습 효과 거의 없음). 답 스타일이 LLaVA GT의 짧은 답과 엇나감 — 본 학습 시 스타일 정렬 기대.

### 6.5 더 많은 샘플 / 다른 step 돌리기

- `--n 40` 으로 val 전부, 또는 `--val-jsonl` 을 train jsonl 로 바꿔 train 세트에 대한 오버핏 체크도 가능.
- 다른 step 쓰려면 `step_N_merged/` 를 따로 만들어 두기 (merged 25 GB × N 개라 디스크 주의). 또는 매번 merge 스크립트로 덮어쓰기.
- 평가 자동화가 필요하면 출력 jsonl 위에 BLEU / ROUGE / VQA metric 얹으면 됨 (별 워크스트림).

### 6.6 vLLM 없이 빠르게 한 장만 — quick_test_image.py

`step_10_merged/` (또는 `base model 디렉토리`) 안에 `quick_test_image.py` 가 딸려 나옵니다. HF `AutoModelForCausalLM` 로 1 GPU 추론하는 예제 — 디버깅 용도로 유용.

```bash
docker exec vuvlm-v050 bash -lc '
DT=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
cd /ephemeral/vuvlm/nemo-RL-v0.5.0/results/sft_nemotron_vl_lora/step_10_merged
$DT/bin/python quick_test_image.py --model_path . --device cuda:0 --max_new_tokens 256
'
```

vLLM path 실패했을 때 HF 경로로 분리 검증하는 용도. 단, §7 부록 패치 #5 (modeling.py 의 `image_flags` / `past_key_values` 패치) 필요.

---

## 7. 부록 — 적용된 vuvlm-local 패치 8개

이 환경이 동작하려면 **clean v0.5.0 위에 다음 8개 패치**가 들어 있어야 합니다. 모두 이미 적용되어 있지만, **클론 새로 만들거나 모델 캐시 날아가면 다시 적용 필요**.

| # | 위치 | 변경 | 이유 |
|---|---|---|---|
| 1 | `nemo_rl/distributed/virtual_cluster.py` | 두 `ray.init()` 모두 `num_cpus=int(os.environ.get("NRL_RAY_NUM_CPUS","64"))` 추가, 두 번째 `ray.init`은 `runtime_env=`/`resources=` kwargs 제거 + `include_dashboard=False` | 호스트 252 CPU 자동검출하면 prestart_python_workers=252 → runtime_env_agent 30s timeout. 또한 두 번째 ray.init에 runtime_env 넘기면 컨테이너에서 raylet의 dashboard/runtime_env agent들이 즉시 죽음 (로그 파일도 안 생김). |
| 2 | worker venv | `pip install "timm<=1.0.22" "open-clip-torch>=3.2.0"` | v0.5.0 `automodel` extra 누락. fork는 megatron-bridge dependency-metadata로 추가했음. Nemotron VL의 RADIO encoder가 timm 필요, modeling code가 open_clip 필요. |
| 3 | `examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml` | `policy.dtensor_cfg.activation_checkpointing: false` | 모델 클래스 `NemotronH_Nano_VL_V2` 가 `gradient_checkpointing_enable()` 미지원 → ValueError. |
| 4 | `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` | `load_base_model()` 호출 직전에 `DefaultLoadPlanner.__init__` monkey-patch (`allow_partial_load=True`) | 모델에 `vision_model.radio_model.summary_idxs` derived buffer 가 있는데 HF safetensors엔 저장 안 돼 있음 → strict load 실패. (검증: 누락 키 정확히 1개, 학습된 weight는 다 있음) |
| 5 | `~/.cache/hf/modules/transformers_modules/nvidia/.../modeling.py` | (a) `image_flags is None` 일 때 `torch.ones((B,))` 로 fallback. (b) `CausalLMOutputWithPast(... past_key_values=getattr(outputs,"past_key_values",None), ...)` (hidden_states / attentions 도 동일) | (a) NeMo-RL 콜레이터가 `image_flags` 안 만듬 → modeling code의 `.squeeze(-1)` 에서 NoneType 에러. (b) Nemotron-H는 Mamba/transformer hybrid라 output 객체에 `past_key_values` attribute 없음. |
| 6 | `examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml` | `policy.dtensor_cfg.lora_cfg.enabled: true` (rest 기본) | full FT는 단일 H100 80GB 한도 초과 (12B BF16 + AdamW FP32 m,v). LoRA로 base frozen → ~24GB. |
| 7 | `nemo_rl/algorithms/sft.py` | line 499, 614: `metrics["global_valid_toks"]` → `metrics.get("global_valid_toks", 0)` | 일부 step에서 키 누락 → KeyError로 학습 중단. fork에 동일 버그 알려져 있음 (`grpo.py` 에서). |
| 8 | `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` + `examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml` | **(code)** `self.model_state_dict_keys = ...` 직후에 `BEGIN/END` sentinel 블록 삽입. 동작: (1) `enable_cpe_support` 모듈의 `checkpoint_seq` 을 monkey-patch 해서 `every=NRL_VISION_AC_UNIT` 주입, (2) RADIO timm VisionTransformer 의 `grad_checkpointing=True` 로 기존 `_forward_cpe` 의 checkpoint 분기 활성화. **(yaml)** `policy.dtensor_cfg.env_vars: {NRL_VISION_AC: "1", NRL_VISION_AC_UNIT: "4"}` 로 워커 env 에 주입. | 패치 #3 로 config-level `activation_checkpointing` 을 끈 상태여서 vision encoder (RADIO ViT) 의 activation 이 계속 살아있음. NemotronH_Nano_VL_V2 가 Automodel `PARALLELIZATION_STRATEGIES` / `VLM_MODEL_CLS_TO_LAYERS` 에 미등록 → default 전략의 AC 루프도 vision 쪽에 안 닿음. `_forward_cpe` 가 이미 `checkpoint_seq(self.blocks, x)` 분기를 가지고 있어서 **구조 변경 없이** (state_dict key 불변 / LoRA target 유지) 그 분기를 활성화 + `every=N` 그룹화. unit=4 기준 32 block → 8 checkpoint group, vision-forward activation ~(N-1)/N 절감. |

### 7.1 패치 적용 자동화

`patches/` 디렉토리 같은 거 만들면 더 좋음 — 지금은 위 표 따라 손으로. 모두 적용된 상태인지 빠르게 검증:

```bash
docker exec vuvlm-v050 bash -c '
echo "[1] virtual_cluster num_cpus: " && grep "NRL_RAY_NUM_CPUS" /ephemeral/vuvlm/nemo-RL-v0.5.0/nemo_rl/distributed/virtual_cluster.py | head -1
echo "[2] timm in worker venv: " && /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python -c "import timm,open_clip; print(timm.__version__, open_clip.__version__)"
echo "[3] activation_checkpointing in config: " && grep "activation_checkpointing" /ephemeral/vuvlm/nemo-RL-v0.5.0/examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml
echo "[4] allow_partial_load patch: " && grep "allow_partial_load" /ephemeral/vuvlm/nemo-RL-v0.5.0/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py | head -1
echo "[5] image_flags fallback in modeling.py: " && grep "image_flags is None" /ephemeral/vuvlm/.cache/hf/modules/transformers_modules/nvidia/NVIDIA*/*/modeling.py
echo "[6] lora_cfg enabled in config: " && grep -A1 "lora_cfg:" /ephemeral/vuvlm/nemo-RL-v0.5.0/examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml | head -2
echo "[7] global_valid_toks defensive get: " && grep "global_valid_toks.*0)" /ephemeral/vuvlm/nemo-RL-v0.5.0/nemo_rl/algorithms/sft.py | head -2
echo "[8] vision AC patch markers: " && grep -cE "^[[:space:]]*# vuvlm-local (BEGIN|END) +\[patch #8" /ephemeral/vuvlm/nemo-RL-v0.5.0/nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py
echo "[inference] peft in dtensor_v2 venv: " && /opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python -c "import peft; print(peft.__version__)"
'
```

> 패치 #8 의 `[8] vision AC patch markers:` 는 **BEGIN 1줄 + END 1줄 = `2`** 가 정상. 다른 숫자 나오면 블록이 중복/누락된 상태.

**Unit 설정 (yaml 경유, 권장)**:

smoke yaml `policy.dtensor_cfg.env_vars:` 블록에 이미 아래 값이 박혀 있습니다 (기본 unit=4):
```yaml
policy:
  dtensor_cfg:
    env_vars:
      NRL_VISION_AC: "1"        # 1=enable / 0=skip
      NRL_VISION_AC_UNIT: "4"   # checkpoint 단위 (blocks / unit)
```
- `NRL_VISION_AC_UNIT`:
  - `"1"` — 32 block 개별 checkpoint (boundary 32개, 최대 메모리 절감, 오버헤드 최대)
  - `"4"` — 4 block 그룹 (boundary 8개) **권장**
  - `"8"` — 8 block 그룹 (boundary 4개, 절감폭 적음)

**전파 경로**: `policy.dtensor_cfg.env_vars` → `nemo_rl/models/policy/lm_policy.py:130` → Ray worker `runtime_env` → `os.environ` inside dtensor_v2 worker → patch #8 이 읽음.

**런타임 검증 (smoke 시작 후 첫 수 초 로그 확인):**
```bash
LOG=$(ls -t /ephemeral/vuvlm/nemo-RL-v0.5.0/logs*/smoke*.log /ephemeral/vuvlm/nemo-RL-v0.5.0/logs*/*.log 2>/dev/null | head -1)
grep "vuvlm-local patch#8" "$LOG"
# 기대 출력 (unit=4):
#   [vuvlm-local patch#8] AC enabled on RADIO ViT: 32 blocks,
#       unit=4 -> 8 checkpoint groups (every=4, grad_checkpointing=True)
# 꺼져 있으면:
#   [vuvlm-local patch#8] SKIPPED vision encoder AC (env NRL_VISION_AC=0)
```

**런타임 토글 (yaml 수정 없이 바꾸기)**:

YAML env_vars 는 CLI override 로 간단히 덮어씀:
```bash
uv run python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_nemotron_nano_v2_llava_smoke.yaml \
  policy.dtensor_cfg.env_vars.NRL_VISION_AC_UNIT=8    # unit 변경
# 또는 꺼버리기
  policy.dtensor_cfg.env_vars.NRL_VISION_AC=0
```
쉘 env var 는 **Ray worker로 전파되지 않음** (lm_policy.py 가 yaml 값만 `env_vars=` 로 넘김). 터미널에서 `export NRL_VISION_AC=0` 하고 실행하면 **아무 효과 없음** — 반드시 yaml 또는 CLI override 로.

**완전 revert (git 없이):**
`nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` 열어서
```
# vuvlm-local BEGIN  [patch #8: vision-encoder activation checkpointing]
```
부터
```
# vuvlm-local END    [patch #8]
```
까지 한 블록 통째로 삭제. 다른 파일 건드릴 것 없음.

### 7.2 처음부터 다시 셋업한다면

```bash
# 0) 컨테이너 down
cd /ephemeral/vuvlm
docker compose -f docker-compose.v050.yml down

# 1) 소스 클론 (이미 있으면 skip)
git clone https://github.com/NVIDIA-NeMo/RL.git /ephemeral/vuvlm/nemo-RL-v0.5.0
cd /ephemeral/vuvlm/nemo-RL-v0.5.0
git checkout v0.5.0
git submodule update --init --recursive

# 2) compose 기동
cd /ephemeral/vuvlm
docker compose -f docker-compose.v050.yml up -d

# 3) 위 7개 패치 모두 적용 (현재 트리에 이미 들어 있으면 #2, #5, #6 만 새로)

# 4) 첫 실행 (uv가 worker venv 만들면서 #2 누락분 깔리는 시점 거기서 timm/open_clip 미설치 에러 → 다시 #2 수동 설치 후 재실행)

# 5) 정상 동작 확인 → 7.1의 verifier 한번

# 6) inference 필요하면 dtensor_v2 venv에 peft 추가 설치 (§6.1)
```

---

## 8. 트러블슈팅 빠른 표

| 증상 | 원인 / 조치 |
|---|---|
| `Failed to register worker to Raylet` 30초 후 | num_cpus 패치 (#1) 안 들어감 |
| `No module named 'timm'` 또는 `open_clip` | worker venv에 deps 누락 (#2) |
| `does not support gradient checkpointing` | activation_checkpointing 끄기 (#3) |
| `Missing key in checkpoint state_dict: vision_model.radio_model.summary_idxs` | DefaultLoadPlanner 패치 (#4) |
| `'NoneType' object has no attribute 'squeeze'` (image_flags) | modeling.py 패치 (#5a) |
| `'NemotronHCausalLMOutput' object has no attribute 'past_key_values'` | modeling.py 패치 (#5b) |
| `CUDA out of memory` Step 1 | LoRA 활성화 (#6) 또는 batch/seq 줄이기 |
| `KeyError: 'global_valid_toks'` 학습 중간 | sft.py defensive get (#7) |
| NCCL `peer access is not supported` (multi-GPU 시도 시) | brev 호스트 IOMMU/ACS — 학습은 single GPU만 검증. inference vLLM 은 한 NVLink pair 안 TP=2 까지 OK, pair-간 (예: 0+2) 은 동일 증상. |
| (inference merge) `ImportError: Found an incompatible version of torchao. Found version 0.14.1, but only versions above 0.16.0 are supported` | `peft/tuners/lora/torchao.py` 의 dispatch 호출 — §6.2 의 `is_torchao_available=lambda: False` monkey-patch 필요 (merge_lora 스크립트에 이미 들어 있음). torchao >=0.16 올리거나 peft downgrade 로도 해결 가능. |
| (inference vLLM) `NCCL CUDA error 217` with TP>=4 | pair-간 연결 — TP=2 로 내리고 한 pair 안 (0,1 / 2,3 / 4,5 / 6,7) 에만 올려라. |
| (inference) peft install 후에도 `No module named 'peft'` | `pip install` 만 쓰면 venv 내부 site-packages 에 안 앉음. `--target $DT/lib/python3.12/site-packages --no-deps` 로 명시. |
| (patch #8) vision AC 가 의심될 때 (OOM, 또는 행동 변화) | 1) 로그에 `[vuvlm-local patch#8] AC enabled on RADIO ViT: ...` 있는지. 2) 의심되면 yaml `policy.dtensor_cfg.env_vars.NRL_VISION_AC: "0"` 또는 CLI override 로 비활성화. 3) unit 조정은 `NRL_VISION_AC_UNIT: "1"/"4"/"8"`. 4) 완전 제거는 `BEGIN/END` sentinel 블록 통째 삭제. |
| (patch #8) `NRL_VISION_AC=1` 을 쉘에서 export 했는데 로그에 "SKIPPED" 로 뜸 | 쉘 env var 는 Ray worker 로 전파되지 않음. **yaml `policy.dtensor_cfg.env_vars:` 에 명시하거나 CLI override 로** 지정해야 함. |
| (patch #8) `WARN could not locate enable_cpe_support module` | RADIO 로딩 순서가 달라졌거나 모델 repo 리비전이 바뀜. `sys.modules` 에서 `enable_cpe_support` 찾기 실패 → `every=1` 로 fallback. unit 지정 효과 없음. |

---

## 9. 참고

- 검증 시점 핵심 메모리 항목 (셰어드 메모리, `/ephemeral/vuvlm/memory/`):
  - `project_v050_smoke_setup.md` — 본 세트업 전체 디테일
  - `feedback_check_fork_first.md` — 클린 클론 디버깅 시작 전에 fork의 `vuvlm-local:` 패치부터 grep 하라
  - `reference_brev_pod_2026-04-21.md` — 호스트 스펙
- fork의 같은 자료 (text-only run 등): `/ephemeral/vuvlm/nemo-RL/DOCKER_VUVLM.md`
