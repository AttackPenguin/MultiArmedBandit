Evaluation of this test was done with the following parameters
```python
T = 1000
a = [1,2,0.5,2,9]
b = [3,7,3,4,1]

model = A2C('MlpPolicy',
            env,
            gamma=0.1,
            learning_rate=0.0008,
            n_steps=1,
            vf_coef=0.9,
            verbose=1).learn(T)
```