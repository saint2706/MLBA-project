# 🚀 Google Colab Setup Guide for Game of Thrones AI

## 📋 **COMPLETE BEGINNER'S GUIDE**

This guide will help you run the Game of Thrones AI script generator on Google Colab using their free powerful GPUs (Tesla T4, V100, or A100).

---

## 🎯 **STEP 1: Access Google Colab**

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Sign in**: Use your Google account
3. **Create New Notebook**: Click "New notebook" or upload our provided notebook

---

## 🎯 **STEP 2: Upload Files to Colab**

### **Method A: Upload Individual Files** (Easiest)
1. In Colab, click the **📁 Files** icon on the left sidebar
2. Click **📤 Upload** and select these files from the `colab/` folder:
   - `game_of_thrones_ai_colab.ipynb` (Main notebook)
   - `improved_helperAI.py` (Core functions)
   - `modern_example_usage.py` (Model classes)
   - `Game_of_Thrones_Script.csv` (Training data)

### **Method B: Clone from GitHub** (Advanced)
```python
# Run this in a Colab cell
!git clone https://github.com/your-username/your-repo.git
%cd your-repo/colab
```

---

## 🎯 **STEP 3: Enable GPU Acceleration**

1. In Colab menu: **Runtime** → **Change runtime type**
2. **Hardware accelerator**: Select **GPU**
3. **GPU type**: 
   - **T4** (Free tier, good for testing)
   - **V100** or **A100** (Colab Pro, fastest training)
4. Click **Save**

---

## 🎯 **STEP 4: Run the Training**

### **Quick Start** (2-3 hours training):
1. Open `game_of_thrones_ai_colab.ipynb`
2. Run **Cell 1**: Install dependencies
3. Run **Cell 2**: Import libraries  
4. Run **Cell 3**: Load and analyze data
5. Run **Cell 4**: Start training (50 epochs)
6. Run **Cell 5**: Generate sample dialogue

### **Extended Training** (6-8 hours, better quality):
- In Cell 4, change `NUM_EPOCHS = 200` for maximum quality
- **Warning**: May require Colab Pro for long sessions

---

## 🎯 **STEP 5: Monitor Training Progress**

### **What You'll See**:
```
🎬 Game of Thrones AI Training Started
📊 Using GPU: Tesla T4 (15GB memory)
📈 Epoch 1/50 | Loss: 3.45 | Time: 45s
📈 Epoch 2/50 | Loss: 2.98 | Time: 44s
📝 Sample: <TYRION LANNISTER> The wine helps...
```

### **Training Indicators**:
- ✅ **Loss decreasing**: AI is learning well
- ✅ **Stable GPU memory**: No crashes expected  
- ✅ **Sample quality improving**: Better dialogue each epoch
- ⚠️ **Runtime warnings**: Normal, ignore unless errors

---

## 🎯 **STEP 6: Download Your Trained Model**

### **Save Your Work**:
```python
# Run this in a Colab cell to download your trained model
from google.colab import files
files.download('trained_got_model.pt')
files.download('training_results.png')
```

### **Files You'll Get**:
- `trained_got_model.pt` - Your trained AI model
- `training_results.png` - Training progress graphs
- `sample_dialogues.txt` - Generated Game of Thrones dialogue

---

## 🎯 **STEP 7: Generate New Dialogue**

Once training is complete, you can generate new Game of Thrones dialogue:

```python
# Example usage in Colab
generator.generate_dialogue(
    character="TYRION LANNISTER",
    prompt="The wine helps me",
    length=100
)
```

**Expected Output**:
```
<TYRION LANNISTER> The wine helps me think more clearly about our situation. 
We need a better strategy if we're going to survive what's coming.
```

---

## 🎯 **TROUBLESHOOTING COMMON ISSUES**

### **Problem**: "GPU not detected"
**Solution**: 
1. Runtime → Change runtime type → GPU
2. Runtime → Restart runtime
3. Check with: `!nvidia-smi`

### **Problem**: "Session timed out"
**Solution**:
- Free Colab: 12-hour limit, save checkpoints frequently
- Colab Pro: 24-hour limit, better for long training

### **Problem**: "Out of memory"
**Solution**:
- Reduce batch size: `BATCH_SIZE = 4` (instead of 8)
- Reduce model size: `HIDDEN_DIM = 512` (instead of 768)

### **Problem**: "Files not uploading"
**Solution**:
- Check file size < 25MB each
- Use Google Drive integration for large files
- Ensure stable internet connection

---

## 🎯 **COLAB TIPS & TRICKS**

### **Free Tier Optimization**:
- **Save frequently**: `Ctrl+S` or File → Save
- **Use checkpoints**: Training saves every 10 epochs
- **Monitor usage**: Check Runtime → View usage

### **Colab Pro Benefits**:
- **Faster GPUs**: V100/A100 instead of T4
- **Longer sessions**: 24 hours instead of 12
- **More memory**: 25GB+ RAM instead of 12GB
- **Priority access**: Skip waiting queues

### **Best Practices**:
- **Keep browser open**: Colab may disconnect if idle
- **Use small test first**: Run 5 epochs to verify setup
- **Download results**: Colab storage is temporary

---

## 🎯 **EXPECTED TIMELINE**

### **Free Colab (Tesla T4)**:
- **Setup**: 5-10 minutes
- **50 epochs**: 2-3 hours  
- **200 epochs**: 8-10 hours (may need Pro)

### **Colab Pro (V100/A100)**:
- **Setup**: 5-10 minutes
- **50 epochs**: 45-90 minutes
- **200 epochs**: 3-5 hours

---

## 🎯 **SUCCESS METRICS**

### **You'll Know It's Working When**:
- ✅ GPU detected and being used
- ✅ Loss decreases from ~4.0 to <2.0
- ✅ Generated text becomes readable
- ✅ Character names appear correctly
- ✅ Dialogue sounds like Game of Thrones

### **Final Quality Examples**:
```
<JON SNOW> I never wanted to be king. The North remembers, and so do I.

<TYRION LANNISTER> A wise man once said that a mind needs books like a sword needs a whetstone.

<DAENERYS TARGARYEN> I am the blood of the dragon. I will take what is mine with fire and blood.
```

---

## 🆘 **GET HELP**

### **If You Get Stuck**:
1. **Check error messages**: Read carefully for clues
2. **Restart runtime**: Runtime → Restart runtime  
3. **Try smaller settings**: Reduce epochs/batch size
4. **Ask for help**: Share error messages for assistance

### **Resources**:
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [GPU Usage Tips](https://colab.research.google.com/notebooks/gpu.ipynb)
- [PyTorch Colab Guide](https://pytorch.org/tutorials/beginner/colab.html)

---

## 🎉 **CONGRATULATIONS!**

Once complete, you'll have your own Game of Thrones AI that can generate dialogue in the style of the show! Share your best generated quotes and enjoy your AI's creative writing! ⚔️🐉👑
