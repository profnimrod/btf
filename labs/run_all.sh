#!/usr/bin/env bash
# Run every laboratory end to end on CPU with the synthetic corpus.
#
#   bash labs/run_all.sh            full run   (~30-60 min, real results)
#   bash labs/run_all.sh --smoke    wiring check (~5 min, results meaningless)
#
# Each stage prints the numbers docs/EXPECTED.md tells you to expect.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:-full}"
if [ "$PROFILE" = "--smoke" ]; then
  PRE_STEPS=120;  CKPT_EVERY=60steps;  RESUME_CKPT=step00000060.pt; PRE_CKPT=step00000120.pt
  SFT_EPOCHS=2;   DPO_LIMIT=20;  GRPO_ITERS=6;  EMB_EPOCHS=2;  EMB_R2_EPOCHS=1
  DISTILL_N=40;   DISTILL_EPOCHS=3;  HL=8;  LBL=3;  QAT_STEPS=15
  SOAK_MIN=0.1;   CALIB_N=16
  echo "### SMOKE PROFILE — wiring check only; results are not meaningful ###"
else
  PRE_STEPS=1200; CKPT_EVERY=600steps; RESUME_CKPT=step00000600.pt; PRE_CKPT=step00001200.pt
  SFT_EPOCHS=6;   DPO_LIMIT=80;  GRPO_ITERS=40; EMB_EPOCHS=8;  EMB_R2_EPOCHS=3
  DISTILL_N=200;  DISTILL_EPOCHS=12; HL=30; LBL=8; QAT_STEPS=60
  SOAK_MIN=0.5;   CALIB_N=64
fi

echo "############ SETUP ############"
python3 tests/env_check.py
python3 data/make_synthetic.py --scale small
python3 corpus/build.py --sources data/samples --dedup minhash:0.9 \
    --decontam eval/JUDGED-v1 --out corpus/v1
python3 tok/train.py --corpus corpus/v1 --vocab 4000 --digits split --out tok/domain-32k
mkdir -p /tmp/gen && cp corpus/v1/text/GEN-*.txt /tmp/gen/ 2>/dev/null || true
python3 tok/train.py --corpus /tmp/gen --vocab 2000 --digits keep --out tok/general
python3 tok/fragmentation.py --tok tok/domain-32k --ref tok/general --terms eval/domain-terms.txt

echo "############ LAB B: pretrain a small domain model ############"
python3 train/pretrain.py --config train/cfg-lab.yaml --data corpus/v1 \
    --max-steps "$PRE_STEPS" --ckpt-every "$CKPT_EVERY" --ckpt-dir ckpt/lab \
    --eval-heartbeat eval/canary-prompts.jsonl --device cpu --log logs/lab.jsonl
python3 train/verify-resume.py --ckpt "ckpt/lab/$RESUME_CKPT" \
    --config train/cfg-lab.yaml --data corpus/v1 --log logs/lab.jsonl --steps 3
mkdir -p /tmp/heldout && cp corpus/v1/text/PROC-2*.txt /tmp/heldout/
python3 eval/bpb.py --ckpt "ckpt/lab/$PRE_CKPT" --tok tok/domain-32k \
    --text /tmp/heldout --label lab-model
python3 train/model.py train/cfg-125m.yaml

echo "############ LAB C: fine-tune and preference-tune ############"
python3 tests/template_parity.py --tokenizer tok/domain-32k
python3 tests/packing_inspect.py
python3 data/audit-sample.py --data data/sft-v1/train.jsonl
python3 train/sft_local.py --base "ckpt/lab/$PRE_CKPT" --data data/sft-v1/train.jsonl \
    --lora-r 16 --epochs "$SFT_EPOCHS" --lr 2e-3 \
    --oversample "Recommend a MODCOD:6" --out ckpt/sft-adapter.pt
python3 train/dpo_local.py --base ckpt/sft-adapter.pt --beta 0.1 --epochs 1 \
    --limit "$DPO_LIMIT" --out ckpt/dpo-adapter.pt
python3 train/grpo_local.py --base ckpt/dpo-adapter.pt --group 6 \
    --iters "$GRPO_ITERS" --temperature 0.7 --out ckpt/grpo-adapter.pt
python3 quant/merge.py --adapters sft,dpo,grpo --out ckpt/merged.pt   # final stack; each adapter bound to this base
python3 eval/harness.py --ckpt ckpt/merged.pt --name parent --limit "$HL" --lb-limit "$LBL"
python3 eval/guardrails.py --ckpt ckpt/merged.pt --both-directions

echo "############ LAB D: embedder and retrieval ############"
python3 pairs/mine.py --corpus corpus/v1/text --out data/pairs-mined.jsonl
python3 pairs/synth.py --corpus corpus/v1/text --decontam eval/JUDGED-v1 \
    --out data/pairs-synth.jsonl
python3 pairs/dedup.py data/pairs-mined.jsonl data/pairs-synth.jsonl \
    --semantic 0.95 --out data/pairs-v1.jsonl
python3 index/build.py --src corpus/v1/text --bm25 --out run/index/baseline
python3 eval/retrieval.py --index run/index/baseline --mode bm25 --k 10
python3 train/embedder.py --pairs data/pairs-v1.jsonl --epochs 0 --out ckpt/emb-untuned.pt
python3 index/build.py --src corpus/v1/text --dense ckpt/emb-untuned.pt --out run/index/dense0
python3 eval/retrieval.py --index run/index/dense0 --model ckpt/emb-untuned.pt --mode dense
python3 train/embedder.py --pairs data/pairs-v1.jsonl --batch 32 --mrl 128,64 \
    --epochs "$EMB_EPOCHS" --lr 1e-3 --out ckpt/emb-r1.pt
python3 index/build.py --src corpus/v1/text --dense ckpt/emb-r1.pt --out run/index/dense1
python3 eval/retrieval.py --index run/index/dense1 --model ckpt/emb-r1.pt --mode dense
python3 pairs/mine-hard.py --model ckpt/emb-r1.pt --pairs data/pairs-v1.jsonl \
    --margin-filter --denoise cross-encoder --out data/hard-negs.json
python3 train/embedder.py --base ckpt/emb-r1.pt --pairs data/pairs-v1.jsonl \
    --hard-negs data/hard-negs.json --batch 32 --epochs "$EMB_R2_EPOCHS" \
    --lr 3e-4 --out ckpt/emb-r2.pt
python3 index/build.py --src corpus/v1/text --dense ckpt/emb-r2.pt --bm25 \
    --acl-metadata --out run/index/v1
python3 eval/retrieval.py --index run/index/v1 --model ckpt/emb-r2.pt --mode dense
python3 eval/retrieval.py --index run/index/v1 --model ckpt/emb-r2.pt --mode hybrid \
    --save eval/reports/retrieval-hybrid.json
python3 eval/pool.py --queries eval/JUDGED-v1/queries.jsonl --index run/index/v1 \
    --systems bm25,dense --model ckpt/emb-r2.pt --out eval/pool.jsonl
python3 eval/judge-sheet.py --pool eval/pool.jsonl --judges 2 --overlap 0.25
python3 train/rerank.py --pairs data/pairs-v1.jsonl --hard-negs data/hard-negs.json \
    --epochs 1 --out ckpt/reranker.pt
python3 eval/bins.py --index run/index/v1 --model ckpt/emb-r2.pt --save eval/reports/bins.json
python3 index/build.py --src corpus/v1/text --dense ckpt/emb-r2.pt --pq m=8 \
    --out run/index/v1-pq
python3 eval/retrieval.py --index run/index/v1-pq --model ckpt/emb-r2.pt --mode dense

echo "############ LAB E: compress, serve, field ############"
python3 calib/build.py --n "$CALIB_N" --len 192 --mix deploy \
    --decontam eval/JUDGED-v1 --out calib/v1
python3 quant/awq.py --native --model ckpt/merged.pt --calib calib/v1 --group 64 \
    --bits 4 --out ckpt/awq-int4.pt
python3 eval/harness.py --ckpt ckpt/awq-int4.pt --name awq-int4 --limit "$HL" --lb-limit "$LBL"
python3 quant/gptq.py --native --model ckpt/merged.pt --calib calib/v1 --group 64 \
    --bits 4 --out ckpt/gptq-int4.pt
python3 eval/harness.py --ckpt ckpt/gptq-int4.pt --name gptq-int4 --limit "$HL" --lb-limit "$LBL"
python3 eval/delta.py --parent parent --children awq-int4,gptq-int4 --gates \
    || echo "[expected] a gate failure here is the lesson (Ch. 14)"
python3 quant/qat.py --master ckpt/merged.pt --fake-quant int4:g64 \
    --data data/sft-v1/train.jsonl --focus "REQ-SC" --steps "$QAT_STEPS" --out ckpt/qat-master.pt
python3 eval/harness.py --ckpt ckpt/qat-master.pt --name qat-master --limit "$HL" --lb-limit "$LBL"
python3 distill/gen.py --teacher ckpt/merged.pt --n "$DISTILL_N" --filter checker \
    --out data/distill-v1.jsonl
python3 distill/train.py --data data/distill-v1.jsonl --student-layers 2 \
    --student-dim 96 --epochs "$DISTILL_EPOCHS" --lr 2e-3 --out ckpt/student.pt
python3 eval/harness.py --ckpt ckpt/student.pt --name student --limit "$HL" --lb-limit "$LBL"
python3 eval/select.py --top 3
python3 eval/mcnemar.py --a eval/reports/retrieval-hybrid.json \
    --b eval/reports/retrieval-hybrid.json
python3 export/gguf.py --model ckpt/qat-master.pt --dtype f16 --out gguf/asst-f16.gguf --name btf-lab
# production path: ./llama.cpp/build/bin/llama-quantize gguf/asst-f16.gguf gguf/asst-q4.gguf Q4_K_M
bash bench/soak.sh --model ckpt/awq-int4.pt --minutes "$SOAK_MIN" --log soak/q4.csv
python3 bundle/seal.py --model ckpt/awq-int4.pt --index run/index/v1-pq \
    --scaffold serve/template.py --out bundle/ops-edge-v1
DG=$(python3 registry/publish.py --artifact ckpt/awq-int4.pt --type model \
    --datasets satcom-sft-v1 --license open | tail -1)
python3 registry/fetch.py --digest "$DG" --verify
python3 registry/bom.py --digest "$DG"
bash deploy/stage.sh --target run/fleet bundle/ops-edge-v1

echo "############ ALL LABS COMPLETE ############"
echo "Compare your numbers against docs/EXPECTED.md."
