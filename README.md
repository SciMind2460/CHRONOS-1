# CHRONOS-1
A small AI model that I am working on, with pre-trained data.


## How do you install Chronos?
```bash
pip install git+https://github.com/SciMind2460/chronos.git
```

(I know it's unnecessarily convoluted, but PyPI is a pain to deal with, and so is Anaconda.)


## How do you use Chronos?
Just use the following command in your Python project:
```python
import chronostextgen
```
The generator command is:
```python
chronostextgen.generator(your_text, max_length=some_random_length, num_return_sequences=1)
```
