

### Am well aware the below CODE / TEXT is Not Python 

```python

$ wget http://images.cocodataset.org/zips/val2017.zip
--2022-10-29 19:02:02--  http://images.cocodataset.org/zips/val2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 52.217.230.121
Connecting to images.cocodataset.org (images.cocodataset.org)|52.217.230.121|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 815585330 (778M) [application/zip]
Saving to: ‘val2017.zip’

val2017.zip                         100%[=================================================================>] 777.80M  4.46MB/s    in 4m 14s  

2022-10-29 19:06:17 (3.06 MB/s) - ‘val2017.zip’ saved [815585330/815585330]

```

```python
$ wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
--2022-10-29 20:09:35--  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 54.231.172.169
Connecting to images.cocodataset.org (images.cocodataset.org)|54.231.172.169|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 252907541 (241M) [application/zip]
Saving to: ‘annotations_trainval2017.zip’

annotations_trainval2017.zip        100%[=================================================================>] 241.19M  1.23MB/s    in 1m 57s  

2022-10-29 20:11:33 (2.05 MB/s) - ‘annotations_trainval2017.zip’ saved [252907541/252907541]


```
~~~
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip === 6 GB -- NOT DONE 
wget http://images.cocodataset.org/zips/unlabeled2017.zip
~~~
