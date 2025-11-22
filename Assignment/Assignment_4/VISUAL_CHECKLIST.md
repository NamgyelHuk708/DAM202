# ✅ VISUAL RUN CHECKLIST - ASSIGNMENT 4

## 🎯 FOLLOW THIS STEP-BY-STEP

```
┌─────────────────────────────────────────────────────────┐
│  ASSIGNMENT 4: T5 TEXT SUMMARIZATION                   │
│  Transformer Decoder-based Sequence Generation         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 1: SETUP                                         │
└─────────────────────────────────────────────────────────┘

Cell 3: Install Packages
├─ Run: !pip install -q transformers datasets...
└─ ✅ Should show: "All packages installed successfully!"

Cell 5: 🔍 ENVIRONMENT VERIFICATION (NEW!)
├─ Check: Python >= 3.8
├─ Check: PyTorch installed
├─ Check: CUDA available
└─ ✅ MUST show: "✅ GPU detected: [GPU Name]"
   ⚠️  If CPU only → Training will take 10-20 hours!

┌─────────────────────────────────────────────────────────┐
│  PHASE 2: DATA LOADING                                  │
└─────────────────────────────────────────────────────────┘

Cells 6-11: Load CNN/DailyMail
└─ ✅ Should show:
   Training:   10,000 samples
   Validation:  2,000 samples
   Test:        2,000 samples

┌─────────────────────────────────────────────────────────┐
│  PHASE 3: TOKENIZATION ⚠️ CRITICAL!                    │
└─────────────────────────────────────────────────────────┘

Cell 15: 🔧 Tokenize Dataset (FIXED!)
├─ Contains: with tokenizer.as_target_tokenizer():
└─ ✅ MUST show: "💡 IMPORTANT: Labels are now properly 
                   formatted for T5 training!"

   ❌ If missing this message → Re-run Cell 15!

┌─────────────────────────────────────────────────────────┐
│  PHASE 4: MODEL LOADING                                 │
└─────────────────────────────────────────────────────────┘

Cell 19: Load T5-small
└─ ✅ Should show:
   Total parameters:     60,506,624
   Model size:           ~231 MB
   Device:               cuda

┌─────────────────────────────────────────────────────────┐
│  PHASE 5: TRAINING SETUP ⚠️ CRITICAL!                  │
└─────────────────────────────────────────────────────────┘

Cell 21: 🔧 Training Configuration (FIXED!)
├─ Check: predict_with_generate=True  ✅
├─ Check: generation_num_beams=4      ✅
└─ ✅ Should show: "Predict with generate: True"

Cell 22: 🔧 Trainer Setup (FIXED!)
├─ Contains: def compute_metrics(eval_pred):
├─ Contains: compute_metrics=compute_metrics
└─ ✅ Should show: "ROUGE metrics will be computed 
                     during evaluation."

Cell 23: 🔍 DATA VALIDATION (NEW!) ⚠️ CRITICAL!
├─ Tests batch structure
├─ Tests forward pass
├─ Tests for NaN loss
└─ ✅ MUST show: "✅ ALL CHECKS PASSED! Data is ready 
                   for training."

   ❌ If shows errors:
   │  ❌ All labels are -100!
   │  ❌ Forward pass produced invalid loss: nan
   │  
   │  → STOP! Re-run from Cell 15!
   │  → Check tokenization has as_target_tokenizer()

Cell 24: 📋 PRE-TRAINING CHECKLIST (NEW!)
├─ 1️⃣  Model on correct device
├─ 2️⃣  Training data exists
├─ 3️⃣  Validation data exists
├─ 4️⃣  Trainer configured
└─ 5️⃣  GPU memory sufficient

   ✅ MUST show: "✅ ALL CHECKS PASSED (5/5)"
                 "🚀 READY TO START TRAINING!"

   ❌ If any check fails → Fix before training!

┌─────────────────────────────────────────────────────────┐
│  PHASE 6: TRAINING ⏰ 60-90 MINUTES                     │
└─────────────────────────────────────────────────────────┘

Cell 25: 🚀 Start Training

✅ GOOD OUTPUT:
┌──────────────────────────────────────────────────────┐
│ [3750/3750 18:43, Epoch 3/3]                         │
│ Epoch  Training Loss  Validation Loss  Rouge1       │
│ 1      1.850         1.650            35.20         │
│ 2      1.420         1.520            36.50         │
│ 3      1.250         1.480            37.80         │
└──────────────────────────────────────────────────────┘

❌ BAD OUTPUT:
┌──────────────────────────────────────────────────────┐
│ [3750/3750 18:43, Epoch 3/3]                         │
│ Epoch  Training Loss  Validation Loss               │
│ 1      0.000         nan                            │
│ 2      0.000         nan                            │
│ 3      0.000         nan                            │
└──────────────────────────────────────────────────────┘

   ⚠️  If you see 0.0 and nan:
   │  → Training is BROKEN!
   │  → Check Cell 23 output
   │  → Re-run from Cell 15
   │  → Verify tokenization fix

Cell 26: Training Analysis
└─ ✅ Should show visualizations and metrics

Cell 27: Save Model
└─ ✅ Should create: ./t5-finetuned-cnn-dailymail/

┌─────────────────────────────────────────────────────────┐
│  PHASE 7: DECODING STRATEGIES                           │
└─────────────────────────────────────────────────────────┘

Cell 31: Strategy 1 - Greedy Decoding
└─ ✅ Generates 10 summaries (fastest)

Cell 33: Strategy 2 - Beam Search (beam=3)
└─ ✅ Generates 10 summaries (better quality)

Cell 34: Strategy 2 - Beam Search (beam=5)
└─ ✅ Generates 10 summaries (even better)

Cell 35: Strategy 2 - Beam Search (beam=10)
└─ ✅ Generates 10 summaries (best quality, slowest)

Cell 36: Strategy 3 - Nucleus Sampling (p=0.9)
└─ ✅ Generates 10 summaries (diverse)

Cell 37: Strategy 3 - Nucleus Sampling (p=0.7)
└─ ✅ Generates 10 summaries (very diverse)

┌─────────────────────────────────────────────────────────┐
│  PHASE 8: COMPARISON & ANALYSIS                         │
└─────────────────────────────────────────────────────────┘

Cells 42-44: Comprehensive Comparison
└─ ✅ Tables, visualizations, analysis

Cells 45-59: Advanced Analysis
└─ ✅ Attention visualization, error analysis

┌─────────────────────────────────────────────────────────┐
│  ✅ FINAL SUBMISSION CHECK                              │
└─────────────────────────────────────────────────────────┘

Before submitting, verify:

[ ] Cell 1: Student name/ID filled in
[ ] Cell 5: GPU was detected and used
[ ] Cell 23: Showed "ALL CHECKS PASSED"
[ ] Cell 25: Training loss < 2.0 (NOT 0.0)
[ ] Cell 25: Validation loss NOT NaN
[ ] Cell 25: ROUGE-1 > 30%
[ ] Cells 31-37: All strategies completed
[ ] Cells 42-44: Comparison analysis run
[ ] Cell 27: Model saved successfully
[ ] All cells have visible output

┌─────────────────────────────────────────────────────────┐
│  🎯 QUALITY BENCHMARKS                                  │
└─────────────────────────────────────────────────────────┘

Minimum (Pass):
├─ Training Loss:    < 2.5
├─ Validation Loss:  < 2.0
├─ ROUGE-1:          > 30%
├─ ROUGE-2:          > 12%
└─ ROUGE-L:          > 22%

Good (B-Grade):
├─ Training Loss:    < 1.5
├─ Validation Loss:  < 1.6
├─ ROUGE-1:          > 35%
├─ ROUGE-2:          > 15%
└─ ROUGE-L:          > 25%

Excellent (A-Grade):
├─ Training Loss:    < 1.2
├─ Validation Loss:  < 1.5
├─ ROUGE-1:          > 38%
├─ ROUGE-2:          > 17%
└─ ROUGE-L:          > 28%

┌─────────────────────────────────────────────────────────┐
│  🆘 QUICK TROUBLESHOOTING                               │
└─────────────────────────────────────────────────────────┘

Problem: NaN loss
├─ Check: Cell 23 shows "ALL CHECKS PASSED"?
├─ Check: Cell 15 has as_target_tokenizer()?
└─ Fix: Re-run from Cell 15

Problem: CUDA out of memory
├─ Cell 21: per_device_train_batch_size=2
└─ Run: torch.cuda.empty_cache()

Problem: Training too slow (>3 hours/epoch)
├─ Check: Cell 5 shows GPU detected?
└─ Check: Cell 24 shows "Model is on cuda"?

Problem: ROUGE scores < 30%
├─ Train longer: num_train_epochs=5
└─ More data: train_subset_size = 20000

┌─────────────────────────────────────────────────────────┐
│  📚 DOCUMENTATION FILES                                 │
└─────────────────────────────────────────────────────────┘

1. QUICK_START_CARD.md          → Read FIRST (2 min)
2. ASSIGNMENT4_FINAL_VERIFICATION.md → Complete guide
3. TRAINING_FIXES.md            → Fix explanations
4. FINAL_SUMMARY.md             → Everything verified

┌─────────────────────────────────────────────────────────┐
│  🎉 YOU'RE READY!                                       │
│                                                         │
│  Follow the cells in order.                            │
│  Watch for ✅ symbols.                                 │
│  Stop if you see ❌ symbols.                           │
│                                                         │
│  Good luck! 🚀                                         │
└─────────────────────────────────────────────────────────┘
```

**Last Updated:** November 22, 2025  
**Status:** ✅ PRODUCTION READY
