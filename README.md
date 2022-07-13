# Isolation Forest
#### It's the code reproduction of the essay: Zhou, Zhi-Hua, etc. Isolation Forest. 2008 Eighth IEEE International Conference in Data Mining. 2008.

Isolation Forest is an algorithm which directly isolates anomalies due to two anomalies' properties: 1. They are the minority in the dataset 
2. Their attributes' values are very different from those of normal instances.

Isolation Forest has the following advantages:
- Isolate anomalies directly
- Only need to build a partial model
- Use random sampling to achieve high accuarcy
- Consume less computational power
- The time complexity and memory usage grows linearly
- Could treat high dimensional data with lots of noise
