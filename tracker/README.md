## implementation Robust scale-adaptive mean-shift for tracking Papers

updating...

### Implementations

________________

This Python code implements a tracking pipeline of Scale Adaptive Mean-Shift method.
_________________

#### TODOS

    - use histMeanShiftAnisotropicScale
    - compare with DAT
    
#### References

- ASMS color tracker

1. Tomas Vojir, Jana Noskova and Jiri Matas, [“Robust scale-adaptive mean-shift for tracking“](http://101.96.10.63/cmp.felk.cvut.cz/~vojirtom/publications/scia2013.pdf). 
    Pattern Recognition Letters 2014.
2. [Vojir C++ implementation](https://github.com/vojirt/asms)
3. [VOT2016 Dataset](http://data.votchallenge.net/vot2016/vot2016.zip)

---

### Setup and Run

#### Environment
- python 3.6
- opencv-python 0.3.1
- fire

#### Setup

```bash
$ pip install -r requirements.txt
```
    
#### Command

```bash
$ python run_color_tracker.py train folder_name
```

---



