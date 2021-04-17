# EM-MIL: Weakly-Supervised Action Localization with Expectation-Maximization Multi-Instance Learning

By Zhekun Luo, Devin Guillory, Baifeng Shi, Wei Ke, Fang Wan, Trevor Darrell, Huijuan Xu (University of California, Berkeley).

### Introduction

Weakly-supervised action localization requires training a model to localize the action segments in the video given only video level action label. It can be solved under the Multiple Instance Learning (MIL) framework, where a bag (video) contains multiple instances (action segments). Since only the bag's label is known, the main challenge is assigning which key instances within the bag to trigger the bag's label. Most previous models use attention-based approaches applying attentions to generate the bag's representation from instances, and then train it via the bag's classification. These models, however, implicitly violate the MIL assumption that instances in negative bags should be uniformly negative. In this work, we explicitly model the key instances assignment as a hidden variable and adopt an Expectation-Maximization (EM) framework. We derive two pseudo-label generation schemes to model the E and M process and iteratively optimize the likelihood lower bound. We show that our EM-MIL approach more accurately models both the learning objective and the MIL assumptions. It achieves state-of-the-art performance on two standard benchmarks, THUMOS14 and ActivityNet1.2.

### License

EM-MIL is released under the MIT License (see the LICENSE file for details).

### Citing EM-MIL

If you find EM-MIL useful in your research, please consider citing:

    @InProceedings{10.1007/978-3-030-58526-6_43,
    author="Luo, Zhekun
    and Guillory, Devin
    and Shi, Baifeng
    and Ke, Wei
    and Wan, Fang
    and Darrell, Trevor
    and Xu, Huijuan",
    title="Weakly-Supervised Action Localization with Expectation-Maximization Multi-Instance Learning",
    booktitle="Computer Vision -- ECCV 2020",
    year="2020",
    }

We build our model based on UntrimmedNet, THUMOS14 and ActivityNet1.2 dataset. Please cite the following papers as well:

Wang, L., Xiong, Y., Lin, D., Van Gool, L.: Untrimmednets for weakly supervised action recognition and detection. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 4325–4334 (2017) 

Jiang, Y.G., Liu, J., Zamir, A.R., Toderici, G., Laptev, I., Shah, M., Sukthankar, R.: THUMOS Challenge: Action Recognition with a Large Number of Classes. http://crcv.ucf.edu/THUMOS14/ (2014) 

Heilbron, F.C., Escorcia, V., Ghanem, B., Niebles, J.C.: ActivityNet: A LargeScale Video Benchmark for Human Activity Understanding. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 961–970 (2015) 

### Preparation:

1. Clone the EM-MIL repository. 
  	```Shell
  	git clone --recursive https://github.com/airmachine/EM-MIL-WeaklyActionDetection.git
  	```

2. Prepare the pretrained extracted features and set the path in `opts.py`.

### Training and Testing:

1. run the following command.

	```Shell
	python run.py
	```