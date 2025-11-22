# 🚀 ASSIGNMENT 4 - QUICK START CARD

## ⚡ TL;DR - What Was Fixed

Your notebook had **training loss = 0.0** and **validation loss = NaN**.

**3 Critical fixes applied:**

1. ✅ **Cell 15:** Added `with tokenizer.as_target_tokenizer():` for T5 labels
2. ✅ **Cell 21:** Changed `predict_with_generate=False` → `True`
3. ✅ **Cell 22:** Added `compute_metrics` function for ROUGE

**3 New safety cells added:**

4. ✅ **Cell 5:** Environment & GPU verification
5. ✅ **Cell 23:** Data validation before training
6. ✅ **Cell 24:** Pre-training final checklist

---

## 📋 RUN ORDER (Critical Cells Only)

### ✅ MUST PASS THESE CHECKS:

| Cell | What to Check | Expected Output |
|------|---------------|-----------------|
| **5** | GPU detected? | `✅ GPU detected: Tesla T4` |
| **15** | Tokenization | `✅ Labels are now properly formatted` |
| **23** | Data validation | `✅ ALL CHECKS PASSED!` |
| **24** | Pre-training | `✅ READY TO START TRAINING!` |

### ⚠️ If ANY check fails, STOP and fix before training!

---

## 🎯 Expected Training Output

### ✅ GOOD (Training is working):
```
Epoch   Training Loss   Validation Loss   Rouge1
1       1.850          1.650             35.20
2       1.420          1.520             36.50
3       1.250          1.480             37.80
```

### ❌ BAD (Training is broken):
```
Epoch   Training Loss   Validation Loss   Rouge1
1       0.000          nan               --
2       0.000          nan               --
3       0.000          nan               --
```

**If you see 0.0 and NaN:** Re-run Cells 15, 21, 22, 23 and check outputs!

---

## 🆘 Emergency Troubleshooting

| Problem | Solution |
|---------|----------|
| **NaN loss** | Check Cell 23 shows "ALL CHECKS PASSED" |
| **0.0 loss** | Re-run Cell 15 (tokenization) |
| **Out of memory** | Cell 21: `per_device_train_batch_size=2` |
| **Too slow** | Check Cell 5 shows GPU detected |

---

## 📊 Assignment Requirements Met

- ✅ Greedy Decoding (Cell 31)
- ✅ Beam Search x3 (Cells 33-35: beam=3,5,10)
- ✅ Nucleus Sampling (Cells 36-37)
- ✅ Comprehensive Comparison (Cells 42-44)
- ✅ Training & Evaluation (Cells 21-27)
- ✅ Analysis & Documentation (Throughout)

---

## ⏱️ Time Estimates

- **Setup:** 5 minutes
- **Data Loading:** 10 minutes
- **Training:** 60-90 minutes (GPU) / 10-20 hours (CPU)
- **Decoding & Analysis:** 30 minutes
- **Total:** ~2-3 hours with GPU

---

## 🎓 Key Difference from Assignment 3

**Assignment 3 (DistilBERT):** Simple classification - labels are just numbers  
**Assignment 4 (T5):** Text generation - labels are token sequences!

**That's why we need `with tokenizer.as_target_tokenizer():` - it's T5-specific!**

---

## ✅ Pre-Submission Checklist

- [ ] Cell 5: GPU detected ✅
- [ ] Cell 23: "ALL CHECKS PASSED" ✅
- [ ] Cell 25: Training loss < 2.0 ✅
- [ ] Cell 25: Validation loss NOT NaN ✅
- [ ] ROUGE-1 > 30% ✅
- [ ] All 3 decoding strategies run ✅
- [ ] Model saved (Cell 27) ✅
- [ ] Name/ID filled (Cell 1) ✅

---

## 🚀 YOU'RE ALL SET!

**Just run cells in order and watch for ✅ checkmarks.**

**If you see ❌ anywhere, stop and investigate!**

Good luck! 🎉
