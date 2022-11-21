1. only support range() syntax
2. does not support input-depend loops, thus not break and continue
3. does not support while syntax
4. logic flow must be in the compile function
5. must trace torch.tensor([a]) instead of torch.tensor(a)
6. a += 1 rewrite to a = a + 1



# File Structure
## ./src
## compile_torchmobile.py
## utils.py
## study_limitation.py
## 