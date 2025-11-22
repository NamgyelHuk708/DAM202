# 🚀 QUICK START GUIDE - Assignment 3

## ⚡ Fast Track (30 seconds)

1. **Open Google Colab** → https://colab.research.google.com
2. **Upload** `Assignment_3_DistilBERT_IMDB.ipynb`
3. **Enable GPU**: Runtime → Change runtime type → T4 GPU
4. **Run All**: Runtime → Run all
5. **Wait**: ~60-90 minutes
6. **Done!** ✅

---

## 📋 What Happens When You Run

### Phase 1: Setup (5 min)
- Installs libraries
- Loads dataset (25k train + 25k test)
- Initializes tokenizer

### Phase 2: EDA (5 min)
- Generates statistics
- Creates visualizations
- Analyzes text/token distributions

### Phase 3: Training (30-45 min) ⏰
- Fine-tunes DistilBERT
- Saves checkpoints
- Monitors progress

### Phase 4: Evaluation (10 min)
- Tests model
- Generates metrics
- Creates confusion matrix

### Phase 5: Analysis (15 min)
- Attention visualizations (10+)
- Error analysis
- Performance comparison

### Phase 6: Export (2 min)
- Saves model
- Generates docs
- Exports results

---

## 🎯 Key Cells to Modify

### Add Your Name
**Cell 1** (Markdown)
```markdown
**Student Name:** [Your Name Here]
```

### Quick Test (Optional)
**Cell 4** - Uncomment for faster testing:
```python
# Use smaller dataset for quick test
small_train_dataset = dataset["train"].select(range(2000))
small_test_dataset = dataset["test"].select(range(500))
```

### Adjust Training Speed
**Cell 8** - Reduce epochs for faster training:
```python
num_train_epochs=2,  # Instead of 3
```

---

## 📊 Expected Outputs

After running, you'll see:

### ✅ Visualizations
- Class distribution bar chart
- Text length histogram
- Word clouds (positive/negative)
- Training curves (loss/accuracy)
- Confusion matrix
- 10+ attention heatmaps
- Multi-layer attention plots
- Token distribution plots
- Performance comparison charts

### ✅ Metrics
- Test Accuracy: ~92-95%
- Test F1-Score: ~92-95%
- Precision/Recall per class
- Classification report

### ✅ Files Generated
- `requirements.txt`
- `README.md`
- `model_results.json`
- `./distilbert_imdb_finetuned/` (saved model)
- `./results/` (training checkpoints)

---

## 🔥 Pro Tips

### Tip 1: Monitor Progress
Watch Cell 9 (training) - you'll see:
```
Epoch 1/3
[████████████████] 100%
Loss: 0.234 | Accuracy: 0.912
```

### Tip 2: Check GPU
After Cell 2, you should see:
```
Using device: cuda
```
If you see `cpu`, enable GPU!

### Tip 3: Save Periodically
After each major phase:
- File → Save a copy
- Or download notebook

### Tip 4: If It Crashes
- Restart runtime
- Run again - checkpoints save progress
- Or use smaller dataset (see Cell 4)

---

## 📱 Running on Mobile/Tablet?

**Not recommended** - Use desktop/laptop for:
- Better visualization viewing
- Easier code editing
- Stable connection
- File management

---

## ⏰ Time Budget

| Task | Time | Can Skip? |
|------|------|-----------|
| Setup | 5 min | ❌ No |
| EDA | 5 min | ⚠️ Optional |
| Training | 30-45 min | ❌ No |
| Evaluation | 10 min | ❌ No |
| Attention Viz | 15 min | ⚠️ Reduce to 5 |
| Documentation | 2 min | ✅ Yes |

**Minimum Runtime**: ~45 minutes (if you skip optional parts)
**Recommended**: ~90 minutes (complete run)

---

## 🆘 Emergency Shortcuts

### If you're REALLY short on time:

1. **Reduce dataset** (Cell 4):
   ```python
   dataset["train"] = dataset["train"].select(range(5000))
   dataset["test"] = dataset["test"].select(range(1000))
   ```

2. **Fewer epochs** (Cell 8):
   ```python
   num_train_epochs=1
   ```

3. **Skip some visualizations**:
   - Comment out Cell 13 (word clouds)
   - Comment out Cell 17 (10 heatmaps)
   - Keep Cell 10 (main evaluation)

**Warning**: This reduces quality but gives you a working submission.

---

## ✅ Submission Checklist

Before you submit:

- [ ] All cells executed successfully
- [ ] Your name in Cell 1
- [ ] Accuracy > 85% (should be 90-95%)
- [ ] All visualizations displayed
- [ ] Model saved (check files panel)
- [ ] Download notebook (.ipynb)
- [ ] Download README.md
- [ ] Download requirements.txt
- [ ] Export notebook as PDF (File → Print → Save as PDF)

---

## 🎯 Grading Points Coverage

| Points | Requirement | Where |
|--------|-------------|-------|
| 15% | EDA | Cells 3-4, 13-14 |
| 20% | Model Implementation | Cells 7-8, 20 |
| 25% | Training & Results | Cells 9-10, 15 |
| 20% | Attention Analysis | Cells 11, 17-18, 23 |
| 10% | Error Analysis | Cell 16 |
| 10% | Documentation | Cells 25-27 |

**Total Coverage**: 100% ✅

---

## 🎉 You're All Set!

**Your notebook has:**
- ✅ 29 cells covering all requirements
- ✅ Professional code quality
- ✅ Comprehensive visualizations
- ✅ Complete documentation
- ✅ Ready to run in Colab

**Just RUN and WAIT!**

Good luck! 🚀

---

**Questions?** Check `NOTEBOOK_GUIDE.md` for detailed explanations.
