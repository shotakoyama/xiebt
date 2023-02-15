# Xiebt

Implementation of Random Noising Xie et al. 2018 (https://aclanthology.org/N18-1057.pdf) on Fairseq.

This is a data augmentation method for grammatical error correction (GEC).

## Install

First, please install fairseq.
I use v0.12.2.

```
pip install fairseq==v0.12.2
```

or

```
git clone https://github.com/pytorch/fairseq.git -b v0.12.2
cd fairseq
pip install -e .
```

If you want to use fairseq v0.10.2, please use code of commit 1db9c58.

Then, please install xiebt.

```
pip install git+https://github.com/nymwa/xiebt
```

or

```
git clone https://github.com/nymwa/xiebt.git
cd xiebt
pip install -e .
```

## Random Noising

First, you train an error generating model.
You can get this by training the reversed GEC (target -> source).

Next, run `fairseq-preprocess` like this.

```
fairseq-preprocess \
	--source-lang src \
	--target-lang trg \
	--testpref monolingual_data.txt \
	--srcdict dict.src.txt \
	--only-source
cp data-bin/dict.src.txt data-bin/dict.trg.txt # This is fairseq's fault.
```

Then, run this.

```
xiebt-generate \
	data-bin \
	--path checkpoint.pt \
	--seed 12345 \
	--beam 4 \
	--max-tokens 6000 \
	--beta-random 8.0
```

As for beta-random, 6.0 is used in Kiyono et al. 2019 (https://aclanthology.org/D19-1119.pdf) and 8.0 in Koyama et al. 2021 (https://aclanthology.org/2021.naacl-srw.16.pdf).
So this example above is not a recommendation of back-translation. If you want to know the optimal condition of back-translation, you have to search it by yourself.

